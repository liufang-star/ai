import getopt
import logging
import os
import sys
import time
import uuid

import openai
from dotenv import load_dotenv
from flask import Flask, request, Response

from data import conversation_service as conv_service
from data import set_database
from util.Result import err, log, ok

app = Flask(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 设置内置日志记录器等级
logger = logging.getLogger('werkzeug')


# logger.setLevel(logging.ERROR)


# code-davinci-002
# text-davinci-003

def get_param(jn, key, default_value):
    return jn[key] if key in jn else default_value


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


@app.route("/stop/chat/<cid>", methods=['PUT'])
def stop_chat(cid):
    logger.info(log(f"id:{cid}"))
    conversation = conv_service.get_by_id(cid)
    conversation["stopGenerating"] = True
    conv_service.save(conversation)
    return ok({})


@app.route("/conv/<cid>", methods=['GET'])
def conv(cid):
    logger.info(log(f"id:{cid}"))
    conversation = conv_service.get_by_id(cid)
    return ok(conversation)


def get_msgs(conversation):
    msgs = [{"role": "system", "content": ""}]
    for c in conversation["convs"]:
        msgs.append({"role": "user" if c['speaker'] == "human" else "assistant",
                     "content": c['speech'] if 'speech' in c else c['speeches'][-1]})
    return msgs


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

    msgs = get_msgs(conversation)
    msgs.append({"role": "user", "content": "为以上对话取一个符合的标题"})
    return Response(generate_chat(cid, request.args, msgs, callback), mimetype='text/event-stream')


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

    msgs = get_msgs(conversation)
    return Response(generate_chat(cid, request.args, msgs, callback), mimetype='text/event-stream')


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


def generate_chat(cid, rjson, msgs, func):
    model = get_param(rjson, "model", 'gpt-3.5-turbo')
    temperature = get_param(rjson, "temperature", .8)
    max_tokens = get_param(rjson, "max_tokens", 1100)
    top_p = get_param(rjson, "top_p", 1.)
    frequency_penalty = get_param(rjson, "frequency_penalty", .5)
    presence_penalty = get_param(rjson, "presence_penalty", 0.0)

    ai_text = ''
    for resp in openai.ChatCompletion.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True):
        delta = resp["choices"][0]["delta"]
        content = delta["content"] if "content" in delta else ""
        ai_text += content
        content = content.replace("\n", "[ENTRY]")
        yield f'data: {content}\n\n'
        conversation = conv_service.get_by_id(cid)
        if conversation["stopGenerating"]:
            break
    yield "data: [DONE]\n\n"
    func(ai_text)


def init_logging():
    logfile_name = './chat.log'
    logging.basicConfig(level=logging.INFO, filename=logfile_name,
                        format="%(asctime)s - [%(levelname)s] %(filename)s$%(funcName)s:%(lineno)d\t"
                               "%(message)s",
                        datefmt="%F %T")


def init_database():
    logging.info(sys.argv[1:])
    opts, args = getopt.getopt(sys.argv[1:], 'u:p:', ["host=", "port=", "databaseName="])
    set_database(opts)


if __name__ == '__main__':
    init_logging()
    init_database()
    app.run(host="0.0.0.0", port=8383, debug=False)
