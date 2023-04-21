# Python Flask AI

>本项目使用了`python flask openai`实现，数据保存在`mongo`中，`davinci`模型，购买的`openai`账号，建立自己的文本模型。
####
## 部署
##### linux部署 CentOS7.5以上版本

####
## clone项目
>git clone https://github.com/liufang-star/ai.git

####
## 前期准备工作（Linux环境下）
### 安装mongodb数据库

| 数据库     | 版本     |
|---------|--------|
| mongodb | 5.0.16 |
[mongodb官网](https://www.mongodb.com/try/download/community)
####
mongodb可以参照[Linux系统下安装MongoDB](https://blog.csdn.net/DecadentF/article/details/119536858?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168196341016800182118768%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168196341016800182118768&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-4-119536858-null-null.142^v85^insert_down1,239^v2^insert_chatgpt&utm_term=linux%E5%AE%89%E8%A3%85mongodb&spm=1018.2226.3001.4187)

### 安装python编译环境

| 编译环境   | 版本     |
|--------|--------|
| python | 3.10.8 |
[python官网](https://www.python.org/downloads/)

### 安装python所需依赖库

| 库             | 版本          |
|---------------|-------------|
| Flask         | 2.2.2       |
| MarkupSafe    | 2.1.1       |
| openai        | 0.25.0      |
| pymongo       | 4.3.3       |
| python-dotenv | 0.21.1      |
| num2words     | 0.5.12      |
| matplotlib    | 3.7.1       |
| plotly        | 5.14.1      |
| scipy         | 1.10.1      |
| scikit-learn  | 1.2.2       |
| pandas        | 2.0.0       |
| titoken       | 0.3.3       |
| numpy         | 1.24.2      |
| transformers  | 4.28.1      |

### 后台运行python
`nohup python3 main.py >/dev/null 2>&1 &`

####
## 使用docker部署
### 修改Dockerfile文件
`CMD ["python","/code/main.py","-u","root","-p","123456","--host","mongo地址","--port","27017","--databaseName","chatui"]`
将mongo准备好，用户账号密码，mongo地址等
### 修改.env文件
`OPENAI_API_KEY=sk-xxx`
使用openai的api-key（需要有openai的账号）
### 使用 docker build 构建image
`docker build -t me/ai-image:v1 .`
### 使用 docker run 创建容器并运行
`docker run -d --name ai -p 8383:8383 me/ai-image:v1`
##
#### 注：csv文件主要是针对我个人业务，csv文件必须包含这五个字段
`'Product', 'KeyMessage', 'TextCategory', 'Doc_Name', 'Doc_Link'`

####
###赞助
如果此项目对你确实有帮助，欢迎给我打赏一杯咖啡～😄
![img.png](img.png)
