
## 指针和引用

+ 指针和引用的区别 **7点** 
+ 参数传递时，什么时候用指针传递，什么时候用引用传递 **3点**

## 堆和栈

+ 二者区别 **3点**

1. 申请方式
2. 申请大小限制
3. 申请效率（栈是系统分配， 堆是程序员分配）

## new/delete 和 malloc/free  
+ 两者区别 **一相同 五不同**
+ new/delete存在的必要性**非基本类型对象的初始化需要调用构造函数和析构函数，malloc/free无法实现调用功能**

## 宏定义 函数  typedef  inline（内联函数） const
+ 宏定义和函数的区别 **3点**
+ 宏定义和typedef的区别  **4点** 类型安全、处理阶段、适用范围、是否为语句
+ 宏定义和inline的区别 **4点** 类型安全、处理阶段、是否是函数
+ 宏定义和const的区别 **3点** 类型检查、处理阶段、是否有分配空间、存放位置（define替换后占用代码段空间、const占用数据段空间）

## 声明 和 定义 的区别 **两点** 是否分配空间、是否可以多次

## strlen 和 sizeof 区别 **三点**
+ 运算符、库函数
+ 适用范围，运算符更广， 库函数只能处理字符串
+ 运算符的结果再编译的时候就能知道，不用用于动态分配

## struct 和 class 的区别 **两同两不同**
+ 同：结构相同、适用范围相同
+ 不同：默认保护等级不同， 默认继承关系不同

## static 的作用 **五点**
+ 不考虑类
1. 隐藏作用
2. 默认初始化为0
3. 延长局部静态变量的生命周期，局部静态变量只初始化一次
+ 考虑类
1. 修饰成员变量  所有类对象共有，必须在类外初始化
2. 修饰成员函数  **没有this指针**无法访问非static成员变量和成员函数

## const 的作用
+ 不考虑类
1. const常量，声明时必须初始化
2. const形参，底层const
+ 考虑类
1. const成员变量，只能通过**构造函数的初始化列表进行初始化**，不能再类声明中进行初始化
2. const成员函数，const对象不用调用**非const成员函数**，非const对象可以；不能改变非mutable的值

## final 和 override overload hide
+ final：该虚函数不能被子类重写
+ override：编译器帮忙检查是否是父类虚函数重写
+ override 和 hide 的区别在于重写的函数是否为虚函数

## 悬空指针 和 野指针

## C++中有哪几种构造函数 **六种**
+ 默认构造函数
+ 初始化构造函数
+ 拷贝构造函数
1. 什么时候使用拷贝构造函数 **三种情况** 形参为对象 用实例化对象去初始另一个对象 函数返回值
+ 移动构造函数
+ 委托构造函数
+ 转换构造函数

## 浅拷贝 和 深拷贝    *是否开辟了新空间*

##  private、protected、public的访问权限和继承关系

## 什么是大小端存储？ 小端：低字节在低位  大端：高字节在低位

## volatile mutable explicit 
+ volatile 每次从内存中读
+ mutable const函数可以改变mutable修饰的值
+ explicit 只修饰类的构造函数，修饰之后不可隐式转换

## C++异常处理
+ try、catch、throw关键字
+ 函数异常声明列表
+ 异常类型 bad_alloc  out of range  bad_typeid  ios_base::failure  bad_cast

## 类型转换
**const_cast、  dynamic_cast、 static_cast**

## new 和 delete 的实现原理
+ 三步  简单类型和复杂类型
1. 简单类型，直接返回开辟空间指针p，直接释放开辟的空间指针p
2. 复杂类型，直接返回开辟空间指针p-4，**delete[]**直接释放开辟空间p-4, **delete**直接释放空间p，这样会报错

## 类成员的初始化方式？ 什么情况下一定要用成员列表初始化？为什么成员初始化列表的方式更快（少调用了一次构造函数）？
**若是函数体里面赋值初始化，在进入之前，会调用一个默认构造函数分配空间，在函数体内，在调用一次拷贝构造函数，总共就调用两次构造函数，所以相对较慢**
+ 两种 **赋值初始化、 成员列表初始化**
+ 四种
1. 初始化引用成员
2. 初始化常量成员
3. 调用父类构造函数，且有相应的参数时
4. 调用成员类构造函数，且有相应的参数时


## 构造函数的执行顺序？  虚基-> 基类 -> 类中的类成员  -> 自己    （父 - 客 - 己）

## C++的string 和 C中的char* 的区别
+ string继承basic_string，string中的属性包含了char* 的属性
+ string可以动态扩容

## 内存泄漏的定义？如何避免（四种方法）？如何检测（Linux - Valgrind工具、 Windows - CRT库）？
+ 避免：计数法、基类析构函数定义虚函数（实现多态析构）、对象数组用delete[]、new/delete和malloc/free成对出现

## 对象复用  && 零拷贝
+ 对象复用：“对象池”，实现对象的重复利用，避免多次创建重复对象，浪费资源
+ 零拷贝：避免CPU将数据从一块存储复制到另一块存储  **push() 和 emplace() 的区别， emplace直接调用构造函数， push调用拷贝构造函数， emplace()是一个零拷贝的例子**

## 移动构造函数  && 拷贝构造函数
+ 对于指针-- 浅拷贝、 深拷贝
+ 形参类型-- 右值引用、 左值引用
+ 避免新的空间分配，降低构造成本

## 静态类型、动态类型   和  静态绑定、动态绑定
**静态类型**：对象在声明时所采用的类型，在编译期已确定
**动态类型**：通常是指一个指针或引用当前所指的类型，在运行期才确定

**静态绑定**：绑定的是静态类型，所对应的函数或属性依赖于对象的**静态类型**，发生在**编译期**
**动态绑定**：绑定的是动态类型，所对应的函数或属性依赖于对象的**动态类型**，发生在**运行期**

## 全局变量 和 局部变量的 区别 （三点）  生命周期、存储位置、作用域

## 指针传递、引用传递的区别（三点）  
+ 任何对引用参数的处理都会用过间接寻址的方式作用到主调函数的相关变量
+ 指针传递本质还是一个值传递

## 类如何实现只能静态分配 或 只能动态分配 ？ 重载运算符new/delete，设为private 、 把构造函数和析构函数设为protected

## 静态成员 和 普通成员的 区别 （四点） 生命周期、共享方式、初始化方式、存储位置

## 消除隐式转换？  用explicit修饰构造函数

## 交换两数，不使用额外空间？ 两种方法（算术、异或）

## 如果有一个空类，会默认添加哪些函数？（四个）

## 友元函数 和 友元类
+ 友元函数可以访问类的私有成员
+ 友元类中的所有成员函数都是类的友元函数

## 介绍一下几种典型的锁
+ 读写锁
+ 互斥锁
+ 条件变量
+ 自旋锁

## this指针 **重点**
