# I/O 文件的读写
import numpy as np
import h5py as h5
from netCDF4 import Dataset
from scipy.io import netcdf
import netCDF4 as netcdf

# | 操作模式 | 具体含义                         |
# | -------- | -------------------------≠------- |
# | `'r'`    | 读取 （默认）                    |
# | `'w'`    | 写入（会先截断之前的内容）       |
# | `'x'`    | 写入，如果文件已经存在会产生异常 |
# | `'a'`    | 追加，将内容写入到已有文件的末尾 |
# | `'b'`    | 二进制模式                       |
# | `'t'`    | 文本模式（默认）                 |
# | `'+'`    | 更新（既可以读又可以写）         |

# 模式 	   r  /  r+	/  w   	w+	 a	  a+
# 读	       +  /	 +	/      	+		  +
# 写		      /  +	/  +   	+	 +	  +
# 创建		    	/  +   	+	 +	  +
# 覆盖		    	/  +   	+
# 指针在开始  + /	 +	/  +   	+
# 指针在结尾		     	         +	  +

# open(file_name [, access_mode][, buffering])


def read_txt(FileName, iCol, iHead, dtype, sep=None):
    global f
    f = None
    try:
        with open(FileName, 'r') as f:
            lines = f.readlines()[iHead:]
            b = [x.split(sep) for x in lines]
            Data = np.array(b, dtype=dtype)
            return Data[:, iCol - 1]
    except FileNotFoundError:
        print('无法打开指定的文件: {}'.format(filename))
    except LookupError:
        print('指定了未知的编码!')
    except UnicodeDecodeError:
        print('读取文件时解码错误!')
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print('结束')


def read_txt1(filename, icol, ihead, dtype, sep=None):
    try:
        # Using NumPy's genfromtxt for efficiency and simplicity
        data = np.genfromtxt(filename, dtype=dtype, delimiter=sep,
                             skip_header=ihead, usecols=icol-1)
        return data
    except FileNotFoundError:
        print('无法打开指定的文件: {}'.format(filename))
    except LookupError:
        print('指定了未知的编码!')
    except UnicodeDecodeError:
        print('读取文件时解码错误!')
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print('结束')


def read_hdf5(file_path, dataset_names):
    datasets = []
    with h5.File(file_path, 'r') as h5_file:
        for name in dataset_names:
            print('Dataset name:', name)
            try:
                data = h5_file[name][()]
                datasets.append(data)
            except KeyError:
                print(f"Dataset {name} not found in the file.")
    return np.array(datasets)


def create_hdf5(file_name, data_dict):
    """
    Create an HDF5 file and write datasets.

    :param file_name: Name of the HDF5 file to create.
    :param data_dict: Dictionary containing dataset names as keys and NumPy arrays as values.
    """
    with h5.File(file_name, 'w') as h5_file:
        for dataset_name, data in data_dict.items():
            h5_file.create_dataset(dataset_name, data=data)


def create_hdf5_with_attributes(file_name):
    with h5.File(file_name, 'w') as hdf5_file:
        # 创建数据集
        dataset = hdf5_file.create_dataset('dataset_name',
                                           data=np.random.rand(10, 10))

        # 添加局部属性到数据集
        dataset.attrs['unit'] = 'k'
        dataset.attrs['description'] = 'Random data'

        # 添加全局属性到文件
        hdf5_file.attrs['title'] = '2023 graduate program'
        hdf5_file.attrs['author'] = 'Yang'

    print(f"HDF5 file '{file_name}' created with datasets and attributes.")


def read_hdf5_attributes(file_name, dataset_name=None):
    with h5.File(file_name, 'r') as hdf5_file:
        # 读取全局属性
        global_attributes = dict(hdf5_file.attrs)

        # 读取局部属性
        local_attributes = {}
        if dataset_name:
            dataset = hdf5_file[dataset_name]
            local_attributes = dict(dataset.attrs)

    return global_attributes, local_attributes


def read_nc(s_file, dataset_names, byte_order='littleendian'):
    r_var = []
    try:
        with netcdf.Dataset(s_file, 'r') as nc:
            for x in dataset_names:
                data = nc.variables[x][()]
                # 检查是否需要转换字节序
                if byte_order.lower() == 'bigendian':
                    # 如果指定为 big endian，则进行转换
                    data = data.byteswap().newbyteorder()
                r_var.append(data)
    except Exception as e:
        print(f"读取文件错误: {e}")
    return r_var

# # Little Endian
# # 定义：在 little endian 字节序中，多字节数据的最低有效字节存储在最低的地址（即开始的位置）
# ，而最高有效字节存储在最高的地址（即结束的位置）。
# # 举例：假设有一个 16 位的数字 0x1234（十六进制），
# 在 little endian 字节序中，它将被存储为 34 12（先低字节后高字节）。

# # Big Endian
# # 定义：在 big endian 字节序中，情况恰好相反。最高有效字节存储在最低的地址，而最低有效字节存储在最高的地址。
# # 举例：同样的数字 0x1234，在 big endian 字节序中，将被存储为 12 34（先高字节后低字节）。


# # 在代码中，检查 byte_order 参数，如果它被设置为 'Bigendian'，则调用 byteswap().newbyteorder() 来改变数组中数据的字节序。
# 这是因为不同的系统或文件格式可能会使用不同的字节序来存储数据，因此在读取时需要进行相应的转换以确保数据的正确性。

# # 为何重要：不同的计算机架构可能会使用不同的字节序。例如，Intel x86 架构是 little endian，
# 而许多网络协议（包括 TCP/IP）使用 big endian 作为标准。
# 因此，处理字节序在读取或交换数据时非常重要，以确保数据的一致性和正确解释。


if __name__ == "__main__":
    data_to_store = {
        "lat": np.array([-1, -2, -3, -4, -5]),
        "lon": np.array([1, 2, 3, 4, 5]),
    }
    file_name1 = "./data/example.hdf5"
    create_hdf5(file_name1, data_to_store)
    data = read_hdf5("./data/example.hdf5", ['lat', 'lon'])

    file_name2 = './data/test.hdf5'
    create_hdf5_with_attributes(file_name2)
    global_attrs, local_attrs = read_hdf5_attributes(
        file_name2, 'dataset_name')

    a = read_txt('./data/Mean1.txt', 1, 1, 'float', ',')
    print('a=', a)
    b = read_txt1('./data/Mean1.txt', 1, 1, 'float', ',')
    print('b=', b)
