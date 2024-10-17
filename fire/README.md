bump文件夹存放的是bump区域的PDE，其中bumpkpoi.py求解的是poisson问题，bumpkflow.py求解的是kflow问题，bump.geo是区域的剖分，详细说明参考https://blog.csdn.net/forrestguang/article/details/124552846

interface文件夹存放的是矩形区域的interface问题，test1.geo是区域剖分，详细说明参考https://blog.csdn.net/forrestguang/article/details/124552846

NS存放的是三种不同的区域下NS方程的代码，详细说明参考https://blog.csdn.net/forrestguang/article/details/124604503

channel存放的是管道流问题代码，详细说明参考https://blog.csdn.net/forrestguang/article/details/124604503

上面提到的.geo文件需要下载gmesh打开，借助gmesh生成剖分文件，借助firedrake解析求解以后，需要下载paraview做可视化处理，firedrake的下载安装可以参考https://blog.csdn.net/forrestguang/article/details/124162872