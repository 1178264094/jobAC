// 快排
void quickSort(vector<int>& a, int l ,int r)
{
    if(l >= r) return;
    int i = l - 1, j = r + 1, x = a[l + r >> 1];
    while(i < j)
    {
        do i++; while(a[i] < x);
        do j--; while(a[j] > x);
        if(i < j) swap(a[i], a[j]);
    }
    quickSort(a, l, j);
    quickSort(a, j + 1, r);
}

// 归并排序
void mergeSort(vector<int>& a, vector<int>& tmp, int l, int r)
{
    if(l >= r) return;
    int mid = l + r >> 1;
    mergeSort(a, tmp, l, mid);
    mergeSort(a, tmp, mid + 1, r);
    int i = l, j = mid + 1, k = 0;
    while(i <= mid && j <= r)
    {
        if(a[i] <= a[j]) tmp[k++] = a[i++];
        else tmp[k++] = a[j++];
    }    
    while(i <= mid) tmp[k++] = a[i++];
    while(j <= r) tmp[k++] = a[j++];

    for(int i = l, j = 0; i <= r; i++, j++) a[i] = tmp[j];
}

// 堆排序
void down(vector<int>& a, int u)
{
    int len = a.size();
    int maxIndex = u;
    if( 2 * u + 1 < len && a[ 2 * u + 1] > a[maxIndex] ) maxIndex = 2 * u + 1;  // 左孩子
    if( 2 * u + 2 < len && a[ 2 * u + 2] > a[maxIndex] ) maxIndex = 2 * u + 2;  // 右孩子
    if(maxIndex != u)
    {
        swap(a[u], a[maxIndex]);
        down(a, maxIndex);
    }
}

void up(vector<int>& a, int u)
{
    while( u && a[u] > a[( u - 1 ) / 2] )
    {
        swap(a[u], a[(u - 1) / 2]);
        u = (u - 1) / 2;
    }
}

void heapSort(vector<int>& a)
{
    for(int i = a.size() / 2; i >= 0; i--)
        down(a, i);
}


// 冒泡排序
void bubbleSort(vector<int>& a)
{
    int len = a.size();
    for(int i = 0; i < len - 1; i++)
    {
        for(int j = 0; i < len - 1 - j; i++)
        {
            if(a[j] < a[j + 1])
            {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// 选择排序
void selectionSort(vector<int>& a)
{
    int len = a.size();
    for(int i = 0; i < len - 1; ++i)
    {
        int min = i;
        for(int j = i + 1; j < len; ++j)
        {
            if(a[j] < a[min]) min = j;
        }
        swap(a[i], a[min]);
    }
}

// 插入排序
void insertionSort(vector<int>& a)
{
    int len = a.size();
    for(int i = 1; i < len; i++)
    {
        if(a[i] > a[i-1])
        {
            int j = i - 1;
            int x = a[i];
            while(j >= 0; x > a[j])
            {
                a[j + 1] = a[j];
                j --;
            }
            a[j + 1] = x;
        }
    }
}

// 计数排序
void CountSort(vector<int>& vecRaw, vector<int>& vecObj)  // 计数排序
{
    if(!vecRaw.size()) return;
    int vecCountLength = (*max_element(begin(vecRaw), end(vecRaw))) + 1;

    vector<int> vecCount(vecCountLength, 0);
    for(int i = 0; i < vecRaw.size(); i++) vecCount[vecRaw[i]]++;
    for(int i = 1; i < vecCountLength; i++) vecCount[i] += vecCount[i - 1];

    for(int i = vecRaw.size(); i > 0; i--) vecObj[ --vecCount[vecRaw[i - 1]] ] = vecRaw[i - 1];
}

// 并查集
int find(vector<int> & p, int x)
{
    if(p[x] != x) p[x] = find(p, p[x]);
    return p[x];
}

void Union(vector<int>& p, int a, int b)
{
    p[find(p, a)] = find(p, b); 
}

// 01背包问题  （二维数组版本）
void test_2_wei_bag_problem()
{
    vector<int> weight = { 1, 3, 4 };
    vector<int> value = { 15, 20, 30 };
    int bagweight = 4;

    // dp二维数组
    vector<vector<int> > dp(weight.size(), vector<int>(bagweight + 1, 0));

    // 初始化
    for(int i = weight[0]; i <= bagweight; i++) dp[0][i] = value[0];

    // 遍历
    for(int i = 1; i < weight.size(); i++)  // 先物品
    {
        for(int j = 0; j <= bagweight; j++)  // 遍历背包容量
        {
            if(j < weight[i]) dp[i][j] = dp[i - 1][j];
            else dp[i][j] = max(dp[i -1][j], dp[i - 1][j - weight[i]] + value[i]);
        }
    }
    cout << dp[weight.size() - 1][bagweight] << endl;
}

int main()
{
    test_2_wei_bag_problem();
}

// 01背包问题 （一维数组版本）   ----- 遍历顺序有区别
void test_1_wei_bag_problem()
{
    vector<int> weight = { 1, 3, 4 };
    vector<int> value = { 15, 20, 30 };
    int bagweight = 4;

    // dp一维数组
    vector<int> dp(bagweight + 1, 0);
    for(int i = 0; i < weight.size(); i++)  // 物品在外层
    {
        for(int j = bagweight; j >= weight[i]; j--) // 背包容量在内层
        {
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
        }
    }
    cout << dp[bagweight] << endl;
}

int main()
{
    test_1_wei_bag_problem();
}

// 完全背包问题（一维数组版本） -----  遍历顺序无所谓

// 求组合数， 外层先遍历物品， 内层遍历背包容量
// 求排列数， 外层先遍历背包容量， 内层遍历物品

// 如果把遍历nums（物品）放在外循环，遍历target的作为内循环的话，举一个例子：计算dp[4]的时候，结果集只有 {1,3} 这样的集合，不会有{3,1}这样的集合，因为nums遍历放在外层，3只能出现在1后面！
void test_CompletePack()
{
    vector<int> weight = {1, 3, 4};
    vector<int> value = {15, 20, 30};
    int bagweight = 4;
    vector<int> dp(bagweight + 1, 0);
    for(int i = 0; i < weight.size(); i++)  // 先遍历物品，再遍历背包     计算的是组合数
    {
        for(int j = weight[i]; j <= bagweight; j++)
        {
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
        }
    }
    cout << dp[bagweight] << endl;
}
int main()
{
    test_CompletePack();
}

void test_CompletePack() {
    vector<int> weight = {1, 3, 4};
    vector<int> value = {15, 20, 30};
    int bagWeight = 4;

    vector<int> dp(bagWeight + 1, 0);

    for(int j = 0; j <= bagWeight; j++) { // 遍历背包容量        计算的是排列数
        for(int i = 0; i < weight.size(); i++) { // 遍历物品
            if (j - weight[i] >= 0) dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
        }
    }
    cout << dp[bagWeight] << endl;
}
int main() {
    test_CompletePack();
}


// 操作系统

// 进程互斥

// （整型信号量）   -- 不满足 “让权等待”，会发生 “ 忙等 ” 
int S = 1;   // 初始化整型信号量，表示当前系统中可用的打印机资源数

void wait(int S)   // wait 原语 （P 操作） ， 相当于 “ 进入区 ”  （整个过程不会被中断）
{
    while(S <= 0); // 如果资源数不够， 就一直等待
    S = S - 1;  // 如果资源数够，则占用一个资源
}
void signal( int S )   // signal 原语  （V 操作），相当于 “ 退出区 ”
{
    S = S + 1;  // 使用完资源后，在退出区释放资源
}

void process0
{
    ...
    wait(S);    // 进入区，申请资源
    使用打印机资源...  // 临界区，访问资源
    signal(S);   // 退出区，释放资源
    ...
}

void process1
{
    ...
    wait(S);    // 进入区，申请资源
    使用打印机资源...  // 临界区，访问资源
    signal(S);   // 退出区，释放资源
    ...
}

void process2
{
    ...
    wait(S);    // 进入区，申请资源
    使用打印机资源...  // 临界区，访问资源
    signal(S);   // 退出区，释放资源
    ...
}


// 记录型信号量  --- -- 满足 “让权等待”，不会发生 “ 忙等 ” 

/* 记录型信号量的定义 */  
typedef struct 
{
    int value;  // 剩余资源数
    struct process *L;  // 等待队列， 用于链接所有等待资源的进程
}semaphore;

/* 某进程需要使用资源时，通过wait原语申请 */
void wait(semaphore S)
{
    S.value--;
    if( S.value < 0 )
    {
        block (S.L);  // 剩余资源不够，使用block原语，让该进程阻塞，并挂到信号量S的等待队列，从而让出cpu
    }
}
/* 某进程使用完资源后，通过signal原语释放 */
void signal(semaphore S)
{
    S.value++;
    if(S.value <= 0)
    {
        wakeup(S.L); // 释放资源后，若还有别的进程在等待资源，使用wakeup原语在等待队列中唤醒一个进程。
    }
}


// 信号量机制实现   进程互斥

semaphore mutex = 1;

process1
{
    ...
    P(mutex); // wait(mutex)
    临界区代码段...
    V(mutex); // signal(mutex)
}

process2
{
    ...
    P(mutex); // wait(mutex)
    临界区代码段...
    V(mutex); // signal(mutex)
}

// 进程同步 : 要让各并发进程按要求有序地推进

若process2的“代码4”要基于process1的“代码1”和“代码2”的运行结果才能执行，那么我们就必须保证“代码4”一定是在“代码2”之后才会执行
process1
{
    代码1;
    代码2;
    代码3;
}

process2
{
    代码4;
    代码5;
    代码6;
}

// 实现  设置同步信号量为S，初始化为0， 在 “前操作” 执行完之后，执行V操作释放资源， 在 “后操作” 执行之前， 执行P操作，这样即使先走到这，也没有资源释放，进程阻塞

semaphore S = 0;

process1
{
    代码1;
    代码2;
    V(S);  // 执行完1，2才释放资源
    代码3;
}

process2
{
    P(S); // 即使先执行到这里，也得等1，2执行放释放资源才能执行，进程在这里阻塞
    代码4;
    代码5;
    代码6;
}

// 信号量机制实现前驱关系

实现思路： 其实每一对前驱关系都是一个进程同步问题（需要保证一前一后的操作）
因此：
1、要为每一对前驱关系各设置一个同步变量
2、在 “前操作” 之后对相应的同步变量执行V操作
3、在 “后操作” 之前对相应的同步变量执行P操作


// 生产者-消费者  问题  （一条边就是一个同步关系， 双向边就是两个同步关系）

-------------  实现互斥的P操作 一定 要在实现同步的P操作之后  -----------------  否则死锁， 连个p操作位置互换，会死锁

（1）问题描述
系统中有一组生产者进程和一组消费者进程，生产者进程每次生产一个产品放入缓冲区，消费者进程每次从缓冲区中取出一个产品并使用。(注: 这里的“产品”理解为某种数据)
生产者、消费者共享一个初始为空、大小为n的缓冲区。
只有缓冲区没满时，生产者才能把产品放入缓冲区，否则必须等待。
只有缓冲区不空时，消费者才能从中取出产品，否则必须等待。
缓冲区是临界资源，各进程必须互斥地访问。

实现代码：

semaphore mutex = 1; // 互斥信号量，实现对缓冲区的互斥访问
semaphore empty = n; // 同步信号量，表示空闲缓冲区的数量
semaphore full = 0;  // 同步信号量，表示产品的数量，也即非空缓冲区的数量

producer()
{
    生产一个产品；
    P(empty);  // 空闲缓冲区的数量 - 1
    P(mutex);  // 上锁访问 缓冲区
    产品放入缓冲区;
    V(mutex);
    V(full);  // 非空缓冲区的数量 + 1
}

consumer()
{
    P(full);  // 非空缓冲区的数量 - 1, 看是否有资源， 实现“先生产， 后消费”的同步关系
    P(mutex); 
    产品读出缓冲区;
    V(mutex);
    V(empty)  // 实现“有空缓冲区，才能生产”的同步关系
}

// 读者-写者问题

要求：
1、允许多个读者可以同时对文件执行读操作
2、只允许一个写者往文件中写信息
3、任一写者在完成写操作之前不允许其他读者或写者工作
4、写者执行写操作前，应让已有的读者和写者全部退出

问题分析：
两类进程：写进程、读进程
互斥关系：写进程 - 写进程、 写进程 - 读进程。 读进程和读进程之间不存在互斥问题

1、写者进程和任何进程都互斥，设置一个互斥信号量rw，在写者访问共享文件前后执行P、V操作。
2、读者进程和写者进程也要互斥，因此读者访问共享文件前后也要对rw执行P、V操作。

重点理解：如果所有读者进程在访问共享文件之前都执行P(rw)操作，那么会导致各个读进程之间也无法同时访问文件
本类问题的关键是如何解决上述问题？

P(rw) 和 V(rw)其实就是对共享文件的“加锁”和“解锁”。既然各个读进程需要同时访问，而读进程与写进程之间又必须互斥访问，那么
可以让第一个访问文件的读进程 “加锁”， 让最后一个访问完文件的读进程 “解锁”。 可以设置一个整数变量count来记录当前有几个读进程在访问文件

实现：
（1）给count加mutex互斥访问  （读者优先）

semaphore rw = 1;  // 用于实现对文件的互斥访问。表示当前是否有进程在访问共享文件
int count = 0;  // 用于记录当前有几个读进程在访问文件
semaphore mutex = 1;    // 用于保证对count变量的互斥访问

writer()
{
    while(1)
    {
        P(rw);  // 写之前 “ 加锁 ”
        ...write;
        V(rw);  // 写之后 “ 解锁 ”
    }
}

reader()
{
    while(1)
    {
        p(mutex);  // 各读进程互斥访问count
        if(count == 0) 
            P(rw);  // 第一个读进程负责“加锁”
        count ++;   // 访问文件的读进程数 + 1
        V(mutex);

        ... read;

        P(mutex);   // 各读进程互斥访问count
        count --;   // 访问文件的读进程数 - 1
        if(count == 0)
            V(rw);  // 最后一个读程序负责“解锁”
        V(mutex);
    }
}

（2）加一个w实现 “ 读写公平法 ”

1、在上面的算法中，读进程是优先的，即当存在读进程时，写操作将被延迟，且只要有一个读进程活跃，随后而来的读进程都将被允许访问文件。
这样的方式会导致写进程可能长时间等待，且存在写进程“饿死”的情况。
2、若希望写进程优先，即当有读进程正在读共享文件时，有写进程请求访问，这时应禁止后续读进程的请求，等到已在共享文件的读进程执行完毕，立即让写进程执行，只有在无写进程执行的情况下才允许读进程再次运行。
为此，增加一个信号量并在上面程序的writer()和 reader()函数中各增加一对PV操作，就可以得到写进程优先的解决程序。


semaphore rw = 1;  // 用于实现对文件的互斥访问。表示当前是否有进程在访问共享文件
int count = 0;  // 用于记录当前有几个读进程在访问文件
semaphore mutex = 1;    // 用于保证对count变量的互斥访问
semaphore w = 1;  // 用于实现 “ 写优先 ”

writer()
{
    while(1)
    {
        P(w);
        P(rw);  // 写之前 “ 加锁 ”
        ...write;
        V(rw);  // 写之后 “ 解锁 ”
        V(w);
    }
}

reader()
{
    while(1)
    {
        P(w);  // 读写公平锁，不能无限优先读者
        p(mutex);  // 各读进程互斥访问count
        if(count == 0) 
            P(rw);  // 第一个读进程负责“加锁”
        count ++;   // 访问文件的读进程数 + 1
        V(mutex);
        V(w);

        ... read;

        P(mutex);   // 各读进程互斥访问count
        count --;   // 访问文件的读进程数 - 1
        if(count == 0)
            V(rw);  // 最后一个读程序负责“解锁”
        V(mutex);
    }
}


// 哲学家问题
semaphore chopstick[5] = {1, 1, 1, 1, 1};
semaphore mutex = 1;  // 互斥的取筷子
Pi()   // i号哲学家的进程
{
    while(1)
    {
        P(mutex);
        P(chopstick[i]); // 拿左边筷子
        P(chopstick[(i + 1) % 5]);  // 拿右边筷子
        V(mutex);

        ...吃饭

        V(chopstick[i]);
        P(chopstick[(i + 1) % 5]);

        ...思考
    }

}