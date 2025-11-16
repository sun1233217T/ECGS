import logging
logger = logging.getLogger("main_logger")
try:
    from IPython.core import debugger
    debug = debugger.Pdb().set_trace
except Exception as e:

    debug = lambda:None
debug_after_n_iters = lambda:None

def unique_identifier_decorator(func):
    def wrapper(*args, **kwargs):
        # 获取调用者的堆栈信息
        caller_frame = inspect.stack()[1]  # 1 表示获取调用此函数的上一层函数信息
        filename = caller_frame.filename   # 文件名
        lineno = caller_frame.lineno       # 行号
        func_name = func.__name__          # 函数名
        module = caller_frame.frame.f_globals["__name__"]  # 模块名

        # 生成唯一标识符字符串
        identifier_str = f"{module}:{filename}:{func_name}:{lineno}"
        # 生成哈希值作为唯一标识符
        unique_id = hashlib.sha256(identifier_str.encode()).hexdigest()

        # print(f"Unique identifier: {unique_id}")
        return func(unique_id, *args, **kwargs)
    return wrapper

do_once_dict = {}
@unique_identifier_decorator
def do_once(unique_id, func, *args, **kwargs):
    if unique_id not in do_once_dict:
        do_once_dict[unique_id] = 1
        return func(*args, **kwargs)
    return None

