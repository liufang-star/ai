# -*- coding: utf-8 -*-
import getopt
import logging
import re
import sys
import time
import uuid
import warnings

import numpy as np
import openai
import pandas as pd
import tiktoken as tiktoken
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from dotenv import load_dotenv
from flask import Flask, request, Response, json
from openai.embeddings_utils import get_embedding, cosine_similarity
from typing import Dict, List, Tuple

from data import conversation_service as conv_service
from data import set_database
from util.Result import err, log, ok


app = Flask(__name__)

load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "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"
COMPLETIONS_MODEL = ""
TEXT = ""
openai.api_key = ""
openai.api_base = ""
openai.api_version = ""


# 获取指定csv列数据
df = pd.read_csv("")

df_bills = df[['Product', 'KeyMessage', 'TextCategory', 'Doc_Name', 'Doc_Link']]


# 删除多余的空格并清理标点符号来执行一些轻微的数据清理
def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()

    return s


# 忽略警告，在下面三行代码中会出现pandas警告，因为我们在对一个数据帧（df_bills）的切片进行更改，可能会影响原始数据帧，使用.loc或.copy都不能消除这种警告。
warnings.filterwarnings("ignore")
df_bills.loc[:, "KeyMessage"] = df_bills["KeyMessage"].apply(normalize_text).copy()
# 删除任何对于令牌限制（~2000 个令牌）来说太长的账单
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
df_bills.loc[:, 'n_tokens'] = df_bills['KeyMessage'].apply(lambda x: len(tokenizer.encode(x))).values
df_bills = df_bills[df_bills.n_tokens < 2000]


# 使用文档模型嵌入每个块，搜索列中都有其相应的嵌入向量（curie_search、KeyMessage）
df_bills['curie_search'] = df_bills["KeyMessage"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))


# 通过函数传递将查询嵌入到相应的查询模型，并从上一步中先前嵌入的文档中找到最接近它的嵌入
def search_docs(df, user_query, to_print=True):
    embedding = get_embedding(
        user_query,
        engine="text-embedding-ada-002"
    )

    df["similarities"] = df.curie_search.apply(lambda x: cosine_similarity(x, embedding))

    # 按文本分类分组并对每个组应用余弦相似性
    df_extend = df.groupby("TextCategory").apply(lambda x: x.sort_values("similarities", ascending=False).head(1))

    # 统计分组之后的总条数
    res = df.sort_values("similarities", ascending=False).head(df_extend.shape[0])

    # if to_print:
    #     display(res)

    return df_extend


# 将输入数据中的每行文本嵌入向量进行预处理，方便进行文本匹配和搜索，主要针对字段KeyMessage，存储为字典
def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    return {
        idx: get_embedding(r.KeyMessage, COMPLETIONS_MODEL) for idx, r in df.iterrows()
    }


document_embeddings = compute_doc_embeddings(df)

# 查询第一条数据，计算KeyMessage每一个条目的向量值，并输出
example_entry = list(document_embeddings.items())[0]


# 设置文本换行、语言模型
SEPARATOR = "\n* "
ENCODING = "gpt2"

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"


# 设置温度、最大标记数（回复的最长字符）、模型，针对 CSV 文件查询的限制
COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 1000,
    "model": COMPLETIONS_MODEL,
}

# 设置内置日志记录器等级
logger = logging.getLogger('werkzeug')


# logger.setLevel(logging.ERROR)


def get_param(jn, key, default_value):
    return jn[key] if key in jn else default_value


# 获取uuid
@app.route("/generate/id", methods=['POST'])
def generate_id():
    return ok(str(uuid.uuid4()))


@app.route("/ai/suitable/<cid>", methods=['PUT'])
def response_suitable(cid):
    if not request.json \
            or 'suitable' not in request.json \
            or 'msg_idx' not in request.json \
            or 'idx' not in request.json:
        return err("请求参数缺失！")
    idx = request.json['idx']
    msg_idx = request.json['msg_idx']
    suitable = request.json['suitable']
    logger.info(log(f"id:{cid},idx:{idx},msg_idx:{msg_idx}\nsuitable:{suitable}"))

    conversation = conv_service.get_by_id(cid)
    if conversation is None:
        return err("对话不存在")

    convs = conversation["convs"]
    if len(convs) <= idx \
            or convs[idx]["speaker"] == "human":
        return err("下标有误")

    convs[idx]["suitable"][msg_idx] = suitable
    conv_service.replace(conversation)
    return ok(None)


@app.route("/text/<cid>/<idx>", methods=['PUT'])
def text_change(cid, idx):
    if not request.json \
            or 'prompt' not in request.json:
        return err("请求参数缺失！")
    prompt = request.json['prompt']
    idx = int(idx)
    logger.info(log(f"id:{cid},idx:{idx}\nprompt:{prompt}"))

    conversation = conv_service.get_by_id(cid)
    if conversation is None:
        return err("对话不存在")

    convs = conversation["convs"]
    if len(convs) <= idx \
            or convs[idx]["speaker"] == "ai":
        return err("下标有误")

    convs[idx]["speech"] = prompt
    conv_service.replace(conversation)
    return ok(None)


# 停止响应
@app.route("/stop/chat/<cid>", methods=['PUT'])
def stop_chat(cid):
    logger.info(log(f"id:{cid}"))
    conversation = conv_service.get_by_id(cid)
    conversation["stopGenerating"] = True
    conv_service.save(conversation)
    return ok({})


# 保存uuid
@app.route("/conv/<cid>", methods=['GET'])
def conv(cid):
    logger.info(log(f"id:{cid}"))
    conversation = conv_service.get_by_id(cid)
    return ok(conversation)


# 获取提问数据的用户名role和提问数据content
def get_msgs(conversation):
    msgs = [{"role": "system", "content": ""}]
    for c in conversation["convs"]:
        msgs.append({"role": "user" if c['speaker'] == "human" else "assistant",
                     "content": c['speech'] if 'speech' in c else c['speeches'][-1]})
    return msgs


# 设置标题
@app.route("/chat/title/<cid>", methods=['GET'])
def chat_title(cid):
    logger.info(log(f"id:{cid}"))
    conversation = conv_service.get_by_id(cid)
    if conversation is None:
        return err("对话不存在")

    def callback(ai_text):
        logger.info(ai_text)
        conversation["title"] = ai_text
        conv_service.replace(conversation)

    # 找到第一条用户消息
    first_user_msg = None
    for msg in get_msgs(conversation):
        if msg["role"] == "user":
            first_user_msg = msg
            break

    # 将第一条用户消息传递给回调函数作为标题的输入
    if first_user_msg:
        user_msg = {"role": first_user_msg["role"], "content": first_user_msg["content"]}
        return Response(generate_chat(cid, request.args, [user_msg], callback), mimetype='text/event-stream')
    else:
        return err("未找到用户消息")


# 重新响应
@app.route("/chat/repeat/<cid>", methods=['GET'])
def chat_repeat(cid):
    logger.info(log(f"id:{cid}"))
    conversation = conv_service.get_by_id(cid)
    if conversation is None:
        return err("对话不存在")

    def callback(ai_text):
        logger.info(ai_text)
        conversation["convs"][-1]["speeches"].append(ai_text)
        conversation["convs"][-1]["suitable"].append(0)
        conv_service.replace(conversation)

    # 获取role为"user"的对话内容
    user_msgs = []
    for msg in get_msgs(conversation):
        if msg["role"] == "user":
            user_msg = {"role": msg["role"], "content": msg["content"]}
            user_msgs.append(user_msg)

    # 将对话内容传递给generate_chat函数
    return Response(generate_chat(cid, request.args, user_msgs, callback), mimetype='text/event-stream')


# 提问
@app.route("/chat/<cid>", methods=['GET'])
def chat(cid):
    if not request.args \
            or 'prompt' not in request.args:
        return err("请求参数缺失！")

    prompt = request.args['prompt']
    logger.info(log(f"id:{cid}\nprompt:{prompt}"))
    conversation = conv_service.get_by_id(cid)

    if conversation is None:
        conversation = {"_id": cid, "title": prompt, "convs": []}

    conversation["convs"].append({
        "speaker": "human",
        "speech": prompt,
        "createTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    })

    def callback(ai_text):
        logger.info(ai_text)
        conversation["convs"].append({
            "speaker": "ai",
            "speeches": [ai_text],
            "suitable": [0],
            "createTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })
        conv_service.save(conversation)

    msgs = get_msgs(conversation)

    conversation["stopGenerating"] = False
    conv_service.save(conversation)
    return Response(generate_chat(cid, request.args, msgs, callback), mimetype='text/event-stream')


# 模型输出
def generate_chat(cid, rjson, msgs, func):
    last_message_content = ""
    for msg in tuple(msgs)[::-1]:
        if isinstance(msg['content'], str):
            last_message_content = msg['content']
            break

    query = last_message_content
    chosen_sections = []
    my = search_docs(df_bills, query)
    count = 0
    sz = []
    doc_link = ''
    doc_name = '请参考以下链接：'
    doc_links = []
    for section_index in my:
        if section_index == "KeyMessage":
            for j in my[section_index]:
                chosen_sections.append(SEPARATOR + j.replace("\n*", " "))
        elif section_index == "Doc_Name":
            for j in my[section_index]:
                if j in sz:
                    a = 1
                else:
                    sz.append(j)
                    doc_links.append("<a href='{}'>{}</a>".format(my['Doc_Link'][count], j))
                count = count + 1

    doc_link = "  |  ".join(doc_links)

    answer = answer_query_with_context(query, df, document_embeddings, chosen_sections, doc_name, doc_link)
    ai_text = answer
    func(ai_text)
    content = ai_text.replace("\n", "[ENTRY]")
    yield f'data: {content}\n\n'
    yield "data: [DONE]\n\n"

    conversation = conv_service.get_by_id(cid)
    if conversation["stopGenerating"]:
        yield "data: [DONE]\n\n"
        return

# 构造文本搜索提示
def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings: Dict[Tuple[str, str], np.ndarray],
        chosen_sections: List[str],
        doc_name: str,
        doc_link: str,
        show_prompt: bool = False
) -> str:
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not
    contained within the text below, say "I don't know."\n\nContext:\n """

    # 将chosen_sections拆分成多个子列表
    MAX_CONTEXT_LENGTH = 990
    section_groups = []
    current_group = []
    current_length = 0
    for section in chosen_sections:
        section_length = len(section)
        if current_length + section_length > MAX_CONTEXT_LENGTH:
            section_groups.append(current_group)
            current_group = []
            current_length = 0
        current_group.append(section)
        current_length += section_length
    if current_group:
        section_groups.append(current_group)

    # 对每个子列表进行文本拼接并进行OpenAI API调用
    found_answer = False
    answer = ""
    for section_group in section_groups:
        if found_answer:
            break
        prompt = header + "".join(section_group) + "\n\n Q: " + query + "\n A:"
        response = openai.Completion.create(
            prompt=prompt,
            engine=TEXT,
            **COMPLETIONS_API_PARAMS
        )
        answer = response["choices"][0]["text"].strip(" \n")
        if answer != "I don't know.":
            found_answer = True

    # 拼接所有结果并返回
    if not found_answer:
        doc_link = ''
        doc_name = ''
        search_result = openai.Completion.create(
            prompt=query,
            engine=TEXT,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.5
        )
        answer = search_result.choices[0].text.strip()

    return answer + doc_name + doc_link


# 保存日志文件
def init_logging():
    logfile_name = './chat.log'
    logging.basicConfig(level=logging.INFO, filename=logfile_name,
                        format="%(asctime)s - [%(levelname)s] %(filename)s$%(funcName)s:%(lineno)d\t"
                               "%(message)s",
                        datefmt="%F %T")


# 初始化mongodb数据库
def init_database():
    logging.info(sys.argv[1:])
    opts, args = getopt.getopt(sys.argv[1:], 'u:p:', ["host=", "port=", "databaseName="])
    set_database(opts)


#运行程序
if __name__ == '__main__':
    init_logging()
    init_database()
    app.run(host="0.0.0.0", port=8383, debug=False)


