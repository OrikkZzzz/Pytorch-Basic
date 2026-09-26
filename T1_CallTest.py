class Person:
    def __call__(self, name):
        print("__call__" + "Hello" + name)

    def hello(self, name):
        print("hello" + name)

person = Person()   # person为实例化对象
person("zhangsan")  # '__call__'让实例化对象变得可调用
person.hello("lio")