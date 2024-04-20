from DS import DualSouring
import time

if __name__ == '__main__':

    path_init = "data_frames"
    path_testbeds = "items_list"
    testbed_name = "testbed_rev03_case1.xlsx"

    # чтение товаров из testbed_name
    #ds = DualSouring(path_init, path_testbeds, testbed_name)
    # генерация списка товаров размером size_m
    ds = DualSouring(path_init, path_testbeds, size_m=10)
    start_time = time.time()
    ds.prepare()
    result = ds.run()

    print('total time is %f sec' % (time.time() - start_time))


