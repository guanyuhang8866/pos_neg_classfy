### 给定文本判断情感极性

**环境准备**
```
python3.6
```

**依赖安装**
```
pip install -r requirements.txt
```

**程序运行**
```
python main.py
```

**验证**
```
post请求：http://ip:5000 参数{'content': '太差,脏.差,我住过的最不好的酒店,以后决不住了'}

返回结果：{'result': 0.002186894416809082},可用postman测试

数值越大越接近正向，越小越接近负向
```