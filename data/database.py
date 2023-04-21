import pymongo


class database(object):

    # def __init__(self):
    #     self.username = "admin"
    #     self.password = "123456"
    #     self.host = "127.0.0.1"
    #     self.port = "27017"
    #     self.databaseName = "admin"
    #     self.companyId = None
    #     self.func_dict = {"-u": self.set_username, "-p": self.set_password, "--host": self.set_host,
    #                       "--port": self.set_port, "--databaseName": self.set_database}


    # 初始化数据库，如果数据库设置了账号密码请使用上面这段代码
    def __init__(self):
        self.host = "124.221.189.58"
        self.port = "27017"
        self.databaseName = "admin"
        self.companyId = None
        self.func_dict = {"--host": self.set_host,
                          "--port": self.set_port, "--databaseName": self.set_database}

    def set_username(self, value):
        self.username = value

    def set_password(self, value):
        self.password = value

    def set_host(self, value):
        self.host = value

    def set_port(self, value):
        self.port = value

    def set_database(self, value):
        self.databaseName = value

    def set_company_id(self, value):
        self.companyId = value

    def call(self, opts):
        for opt in opts:
            self.set_key(opt)

    def set_key(self, opt):
        key, value = opt
        if value and isinstance(value, str):
            value = value.strip()
        self.func_dict[key](value)

    # def valid(self):
    #     return self.username is not None and \
    #         self.password is not None and \
    #         self.host is not None and \
    #         self.port is not None and \
    #         self.databaseName is not None

    def valid(self):
        return self.host is not None and \
            self.port is not None and \
            self.databaseName is not None

    # def get_database(self):
    #     conn = pymongo.MongoClient(
    #         f'mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.databaseName}?authSource=admin')
    #     print(conn[self.databaseName])
    #     return conn[self.databaseName]

    # 建立数据库连接
    def get_database(self):
        conn = pymongo.MongoClient(
                f'mongodb://{self.host}:{self.port}/{self.databaseName}')
        print(conn[self.databaseName])
        return conn[self.databaseName]
