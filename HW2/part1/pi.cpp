#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>
using namespace std;
unsigned long long num_cycle=0;//投擲到目標的數量
pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;//初始化mutex

void *compute(void* exe)
{
    unsigned int seed=time(NULL);
    long long int cycle=0;
    long long int exeu=*(long long int*)exe;//exeu為被分配到的執行數量
    for(int i=0;i<exeu;i++)
    {
        double x = ((double)rand_r(&seed) / (RAND_MAX)) * 2 - 1;
        double y = ((double)rand_r(&seed) / (RAND_MAX)) * 2 - 1;
        //double distance_squared = x * x + y * y;
        if (x* x + y * y < 1.f)
            cycle++;
    }

    pthread_mutex_lock(&lock);
    num_cycle+=cycle;
    pthread_mutex_unlock(&lock);

	pthread_exit(NULL);
	//return NULL;
}
int main(int argc,char **argv)
{
    int num_threads=atoi(argv[1]);//string2int，所有thread的數量 

    long long int num_toss=atoll(argv[2]);//string2llint，總共投擲數量

    pthread_t* t = new pthread_t[num_threads];//宣告t為有num_threads個threads的陣列
    
    long long int num_exe=num_toss/num_threads;//每個thread的執行次數
    
    //創造# of threads-1個threads
    for(int i=0;i<num_threads-1;i++)
        pthread_create(&t[i],NULL,compute,(void*)&num_exe);

    num_exe+=num_toss%num_threads;//剩下所有的投擲量
    pthread_create(&t[num_threads-1],NULL,compute,(void*)&num_exe);//最後一個thread

    //等待所有thread做完
    for(int i=0;i<num_threads;i++)
        pthread_join(t[i],NULL);
    delete [] t;

    double pi=4*num_cycle/((double)num_toss);
    cout<<pi<<endl;
    return 0;
}
