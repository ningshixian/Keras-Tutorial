## 标题

> 无序列表

- / - *+
- 星号、加号、减号

> 有序列表

1. ​
2. ​

  > > 嵌套引用
>
> /> 引用

> 分割线

---

> 代码块	缩进

`import numpy`



> 行内链接  [方括号] (地址)

This is [an example](http://www.baidu.com) inline link.

> 参考式链接 [方括号][]

This is [an example][id] inline link.

[id]:http://www.baidu.com	"“百度”"

> 强调

*粗体*  **cuti** <u>下划线</u>

> 图片 
>
> - 一个惊叹号 `!`
> - 接着一个方括号，里面放上图片的替代文字
> - 接着一个普通括号，里面放上图片的网址，最后还可以用引号包住并加上 选择性的 `title` 文字。

![成绩](/users/ningshixian/desktop/成绩.png)

### 数学表达式

要启用这个功能，首先到`Preference`->`Markdown`中启用。然后使用`$`符号包裹Tex命令，例如：`$lim_{x \to \infty} \ exp(-x)=0$`将产生如下的数学表达式：

$lim_{x \to \infty} \ exp(-x)=0​$

输入两个美元符号，然后回车，就可以输入数学表达式块了。例如：

$$\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0 \\\end{vmatrix}$$

使用^表示上标，_表示下标，{}将多个字符表示为一个整体
分数使用\frac{分母}{分子}
开方使用\sqrt[次方]{被开方数}
后附各种符号的对照表……


### 下标

下标使用`~`包裹，例如：`H~2~O`将产生水的分子式。H~2~O

### 上标

上标使用`^`包裹，例如：`y^2^=4`将产生表达式

### 代码

- 使用`包裹的内容将会以代码样式显示，例如

使用`printf()`

### 强调

使用两个*号或者两个_包裹的内容将会被强调。例如

**使用两个*号强调内容**

__使用两个下划线强调内容__

### 表格

| 姓名   |  性别  |  毕业学校  |   工资 |
| :--- | :--: | :----: | ---: |
| 杨洋   |  男   | 重庆交通大学 | 3200 |
| 峰哥   |  男   |  贵州大学  | 5000 |
| 坑货   |  女   |  北京大学  | 2000 |

### 任务列表

使用如下的代码创建任务列表，在[]中输入x表示完成，也可以通过点击选择完成或者没完成。

- [ ] 吃饭
- [ ] 逛街
- [ ] 看电影
- [ ] 约泡

目录大纲自动形成
typora可以通过[TOC]自动根据你的标题生成目录


具体[运算符对照表](http://ididsec.com/articles/3/)


