# import threading
# import importlib

# timer = threading.Timer(1, lambda: None)

# try:
#     # 启动定时器
#     timer.start()
#     from knn_cuda import KNN
#     # 尝试导入模块
#     # module = try_import_module("KNN")
# finally:
#     # 无论导入操作是否成功，都停止定时器
#     timer.cancel()

# print('11')

# from knn_cuda import KNN
# print('11')


import multiprocessing
import importlib

def import_module(queue, module_name, from_import):
    try:
        module = importlib.import_module(module_name)
        if from_import:
            # 如果指定了要从模块中导入特定的成员
            module = getattr(module, from_import)
        queue.put(module)
    except (ImportError, AttributeError) as e:
        queue.put(e)

if __name__ == "__main__":
    # 使用队列在进程间通信
    queue = multiprocessing.Queue()
    # 模块名称和要导入的成员
    module_name = "knn_cuda"
    from_import = "KNN"

    # 创建并启动一个进程来尝试导入模块
    p = multiprocessing.Process(target=import_module, args=(queue, module_name, from_import))
    p.start()

    # 等待3秒
    p.join(3)

    if p.is_alive():
        # 如果进程仍在运行（即导入操作超时），终止它
        p.terminate()
        p.join()
        print(f"Failed to import {from_import} from {module_name} within 3 seconds.")
    else:
        # 尝试从队列中获取导入的模块或捕获的异常
        result = queue.get()
        if isinstance(result, Exception):
            print(f"Error importing {from_import} from {module_name}: {result}")
        else:
            print(f"Successfully imported {from_import} from {module_name}")
            KNN = result  # 这里假设成功导入了KNN
    print('11111')