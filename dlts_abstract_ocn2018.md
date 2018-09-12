% title: Enhancing Support in OpenMP to Improve Data Locality in Application Programs Using Task Scheduling 
% author: Vivek Kale and Martin Kong. 
% date: July 8th, 2018


## Motivation

**We first present the motivation of adding constructs in OpenMP that support improving data locality of tasks scheduled to threads. One of the authors will present the introduction/motivation.**

Data locality is important for efficient execution of an OpenMP application in which work is assigned or scheduled to threads dynamically. Using the clause ‘affinity’ for task scheduling proposed for OpenMP 5.0 can improve data locality~\cite{ompaffclause}. However, the task scheduling strategies are fixed by the runtime system, even with the hints available to the affinity to the affinity clause. While having a few fixed task scheduling scheduling strategies taking into account data locality can be beneficial for most applications-architecture pairs, it is arguable that this small set of task scheduling strategies isn’t beneficial for all applications-architecture pairs~\cite{dynwork2, dynwork, worksteal99,DonfackMulticore, DPLASMA,Kulkarni08schedulingstrategies}. OpenMP needs adequate amount of support to maintain high levels of data locality when scheduling tasks to threads. We need to provide a larger numberof hints to the OpenMP runtime of how to assign OpenMP tasks to threads in a way that preserves data locality. Specifically, we need (a) task-to-thread affinity in OpenMP to reduce capacity cache misses on a multi-core node, which we’ll refer to as locality-awareness, and (b) task-to-thread affinity in OpenMP to reduce coherence cache misses on a multi-core node, which we’ll refer to as locality-sensitivity. Our ideas build on the work on the affinity clause being proposed for addition to OpenMP version 5.0~\cite{OpenMP} in that the user provides input to the clause as hints on \textit{what} data needs to be localized and the \textit{degree} to which the data should be localized. In this work, our contribution is the addition of constructs to OpenMP that provides and allow for a rich set of task scheduling schemes having locality-awareness or locality-sensitivity. We focus on developing (a) and building on (b) from previous work.\nocite{Olivier:2012:CMW:2388996.2389085}
**We proceed motivating the need for locality-sensitivity and locality-awareness in OpenMP application programs, suggesting the need to change OpenMP to enhance support or locality-awareness. We talk about two specific proposals for new constructs in OpenMP for enhancing data locality in OpenMP. At the end of each proposal, we briefly talk about the implementation issues for the addition of the constructs and we discuss performance benefits of the approach specified in the proposal. Martin will present the work on locality-awareness, i.e., the content in Section 2. Vivek Kale will present the work relating to locality-sensitivity, i.e., the content in Section 3.** 

## Locality-awareness in OpenMP
###  The Need for Data-driven Task Scheduling 
% Outer worksharing - Inner schedule
Consider the following code for computing the product of two matrices. 

%\markdownInput{listings/matmul.md}

```C++, caption=Matrix Matrix Multiplication
#define M 2048
#define N 4096
#define P 1024
void matmul(int** A, int ** B, int **C)
{
int i, j, k;
for (i = 0; i < M; i++)
  for (j = 0; j < N; j++
    for (k = 0; k < P; k++)
       C[i][j] += A[i][k] * B[k][j];
}
```

% [!code-C[Main](listings/matmul.C "This is source file")]

% [!include[title](myStyle.csl)]

% [1]: http://www.google.com/

% [Google][1]

% <http://www.google.com>

%[https://msdn.microsoft.com/en-us/library/618ayhy6.aspx](https:/% /msdn.microsoft.com/en-us/library/618ayhy6.aspx)

% \begin{figure}[ht!]
% \lstinputlisting[language=C]{./listings/claImpl.C}
% \caption{Caption2}
% \label{fig:my_code}
% \end{figure}

% ![alt text]("Title")

% [I'm a relative reference to a repository file](myStyle.csl)


%\begin{figure}[ht!]
%    \centering
%    \includegraphics{}
%    \caption{Caption}
%    \label{fig:my_label}
% \end{figure}

Using the clause `taskloop`, one can delegate to the OpenMP runtime system how the work will be parallelized across threads without restricting oneself to a static schedule’s assignment of work to threads. However, using taskloop in this context creates spurious memory flushes and cache capacity misses. The user will have ideas on what arrays should be localized, i.e., privatized, through their knowledge of the numerical algorithm. The user can provide hints to the runtime system implementing taskloop so as to reduce cache misses caused by accesses to particular data, e.g., an array used in computation.

### Array Segment Privatization 

The key idea of our solution for supporting locality-awareness in OpenMP programs with work scheduled dynamically is to use an outer loop iterator for worksharing, but schedule them according to some interior loop iterator that has control on the data space being accessed. This is a form of data-driven scheduling.

Consider again the following code for computing the product of two matrices. Array `B` can be partitioned into some number of chunks, where the number of chunks is, e.g., number of cores or worker threads. In this example, we assume we run the code on a node with 4 cores. We can use 4 as the number of chunks to split array `B` into 4 disjoint sets `B[*][0:N/4-1]`, `B[*][N/4:2N/4-1]`, `B[*][N/2:3N/4-1]` and `B[*][3N/4:N-1]`. For convenience, we call these sets `BJ1`, `BJ2`, `BJ3` or `BJ4`.

These memory regions are associated to the memory space accessed by threads 0-3 (or cores 0-3). Threads can only execute on the core that owns a specific subset of `B`. Cyclic shifts are performed so that each thread can access all `BJ`s to complete its computing.

```C++, caption=Matrix Matrix Multiplication with tasking.
#define M 2048
#define N 4096
#define P 1024
void matmul(int** A, int ** B, int **C) {
#pragma omp taskloop workshare(i,4) schedule(j,B,4)
for (i = 0; i < M; i++)
  for (j = 0; j < N; j++)
    for (k = 0; k < P; k++)
       C[i][j] += A[i][k] * B[k][j];
}       
```

The effect of the pragma in the code above is to schedule iterations of the workshared loop on the core that owns a specific subset of `B` (controlled by iterator `j`). In the example above, `B` is selected because the memory traffic induced by writing to array `C` and reading from array `A` is small. Effectively, the value written to entry `C[i][j]` can be computed on a single scalar, whereas for `A`, the memory required is proportional to the value of `P`. If the total size of `B` exceeds the Lowest Level Cache’s capacity, the number of chunks can be chosen so that some number of `BJ` chunks fit in the last level-private cache of a single core, e.g., some large L2 cache. 

To avoid spurious memory flushes and cache capacity misses, we propose extending OpenMP with explicit privatization mechanisms. In the matmul example, array `C` can be classified as a “Write, Shared” memory region, array `A` as “Read, Shared” and array `B` as “Read, Private”. Given that the work sharing is driven by iterator `i`, we can add a clause to compute the values of array `C` in a “row by row” manner, buffering the writes, pushing them to main memory in bulk once the buffer is filled, and at a given time (or iterator). For instance, if we had 4 threads, each with a buffer size of `4N`, we could pipeline the write of `BUF(C,thread)` at times `(i % 4 == thread_id)`, i.e., thread 0 writes its buffer to main memory whenever `i % 4 == 0`, thread 1 writes when `i % 4 == 1`, and so on. With respect to the Outer Worksharing - Inner Scheduling or Array Segment Privatization approach, array `B` would be labeled or tagged as “Read Shared”. This means that the data of array `B` would invariably be accessed by potentially all threads, and that a data-driven scheduling policy could be beneficial if `B` can be split into disjoint data sets. Lastly, depending on the loop order in our matmul example, a segment of array `A` can be pre-loaded and kept in cache, e.g., the L1 cache.


```C, caption=Matrix-Matrix multiplication with tasking using Array Segment Privatization
#define M 256
#define N 1024
#define P 256
void matmul(int** A, int ** B, int **C) {
#pragma omp taskloop workshare(i,4) asp(C:writeshared,i%4) asp(A[0:P]:readshared) asp(B[0:N/4]:readprivate)
for (i = 0; i < M; i++)
  for (j = 0; j < N; j++)
    for (k = 0; k < P; k++)
       C[i][j] += A[i][k] * B[k][j];
}
```


### Performance
This approach will allow to schedule the writes to main memory and reduce the potential contention for bandwidth on the shared memory interconnect that could arise for frequent updates to the same memory location. The important observation here is that ~_suitable mechanisms that allow to associate subregions of a shared array to a core are required to maintain good data locality_~. The user is required to request private regions with specific buffer sizes. Most conditions envisioned for determining when some buffer must be reloaded or dumped can be represented as a function of the worksharing iterator.

## Locality-sensitivity in OpenMP

### Need for Using Task-to-thread Assignment History

The current construct ```taskloop``` in OpenMP, using the ```untied``` modifier for load balancing of tasks across threads or cores, is beneficial for irregular parallel computation such as sparse matrix multiplication\cite{ChapmanOpenMP,OpenMP,Kulkarni08schedulingstrategies}, where no pattern of load imbalance exists across outer iterations. The construct `taskloop` in OpenMP can also be beneficial for application code in which the load imbalance of the computation of the application code exhibits a repeated pattern over application timesteps, i.e., is __persistent__\cite{NamdSC02}. Note that using OpenMP's ```schedule``` clause instead of OpenMP's ```taskloop``` construct would be restricted in its benefits because the pattern of load imbalance and data locality can best be determined and made use of through a task scheduling strategy.


```C, caption= Barnes-Hut code with tasking
Process(void * arg) 
{ 
  register const int slice = (long) arg;
  int tid = (long) arg; 
  int i;
  #pragma omp taskloop grainsize(4)
  for(i=0; i<n; i++)
     body[i]->ComputeForce(groot, gdiameter);
}
```

When the computation of an OpenMP application code is load balanced across cores over invocations of a timestep using a ```taskloop``` construct, the application’s performance degrades when a task is scheduled to a core different than it had been in the previous invocation of the OpenMP computation region. The performance degrades due to a coherence cache miss. The coherence cache miss is caused by the need to move data __from__ the core that ran the task in the previous invocation __to__ the core that is scheduled to run the task in the current invocation. Across cores of a multi-core node, such performance degradation is non-trivial if the cost of moving, or migrating, the data exceeds the benefit of load balancing obtained from migrating the data to another core. Using the affinity clause may reduce cache misses to some extent, but an application programmer knowledgeable of the partitioning of the computation across cores can improve data locality much further through hints to the runtime. The non-trivial degradation can be avoided in OpenMP application code using tasking if the tasks are assigned to threads __carefully__ so as to avoid the coherence cache miss, in particular when the load imbalance of the computation of the application code exhibits a repeated pattern over application timesteps, i.e., is __persistent__~\cite{CharmppOOPSLA93}, and the data that a core uses across outer iterations has a complementary pattern.

%\lstinputlisting{listingsDir/sampleStencil.c}


### Locality-sensitive Task Scheduling

The key idea of our solution for improving locality-sensitivity is to specify a new task scheduling strategy that can be tuned so as to reduce synchronization and scheduling overheads. The description and experimentation for such a scheduling strategy is shown in previous work~\cite{dynwork6}. The runtime system should make data of tasks be reused on a core as much as possible before that data must migrate to another core. The runtime system receives hints from the user on (1) how to determine, based on history, which task a thread should execute next (2) the number of queues to use in order to reduce synchronization costs and (3) the likelihood that the data locality constraint will be ignored and a random task will be stolen from the queue. 

We propose adding a new schedule called constrained locality-sensitivity, or cla, with the modifiers:

- `num_queues_per_team`: Reduces synchronization overheads through through having multiple shared queues. 
- `history`: Specifies the mode, or methodology (from a prespecified set of methodologies) in which history is used to select a task from the shared queue. 
- `constraint_value`: Reduces coherence cache misses by having an adjustable parameter for the probability, between 0.0 and 1.0, that a task is chosen according to history from the previous outer iteration.

The scheduler can balance or tune how much load balance and how much data locality the application code has through the use of the variable randomizationFactor. The constraint allows one to tradeoff the decrease in time of cache misses but larger search time in the queue for the right tasklet with better load balancing from randomization.

The application programmer specifies which data should be associated with one particular thread with id `tid` and what data can be shared across all threads. The application programmer specifies the degree to which a task should be associated with one particular thread with id `tid`. Let’s say that we run the application code below on a node of 4 cores. The number of threads in the team in the parallel region is four. When each of the the four threads each pick up work from the shared work queue, a thread first generates a random number between 0.0 and 1.0 used to determine whether (1) a thread dequeues a task by being sensitive to that task being local or (2) a thread dequeues a random task from the queue. If the number generated determines (1), thread searches for the first task in the queue which has run on that thread in the previous invocation of the taskloop computation region. If the thread finds such a task, the thread dequeues and the executes the task. If the thread doesn’t find such a task, the thread will dequeue the task at the head of the queue. The options passed to the affinity scheduling clause allows one to tune the degree to which load balancing is maintained with respect to data locality.

% TODO: explain the code better. 

```C, caption=Barnes-Hut code with Locality-Sensitive tasking
Process(void * arg) 
{ 
  register const int slice = (long) arg;
  int tid = (long) arg; 
  int i;
  #pragma omp taskloop affinity schedstrat(history(tid):randomizationFactor) grainsize(4)
  for(i=0; i<n; i++)
     body[i]->ComputeForce(groot, gdiameter);
}
```

To implement the idea in an OpenMP runtime, we need to set and follow implementation guidelines, especially since the overheads of the task scheduling or loop scheduling are dependent on the implementation in addition to the application. The implementation needs to ensure that it doesn’t create false sharing in misalignment of shared queue. The implementation should decrease search time when searching for tasks in queue that match the requested thread id  (or, more generally, the requested ‘locality tag’). The implementation should reduce synchronization overheads by supporting multiple shared queues and tuning in parameter of the number of queues. The implementation should use an efficient implementation of work stealing, trying to reduce the cost of contention on the shared memory interconnect. The implementation should ideally have an automatic determination of parameters of the task scheduling strategy, e.g., task size or chunk size. The implementation should support history from previous outer iterations to determine the parameters of the task scheduling strategy in the current outer iteration. 

### Performance

There will be fewer coherence cache misses and less capacity cache misses with more memory bandwidth on the bus. Some benefits not addressed here but that can be addressed are:  (1) The idea won’t decrease synchronization overheads. The prefetching engine still can’t be beneficial for constrained dynamic task scheduling because of the randomized branch involved in the strategy.


## Conclusion 

**We conclude the presentation by explaining how our additions to OpenMP task scheduling will benefit data locality and talk about next steps in moving the proposal forward within OpenMP including our draft schedule / timeline for getting the constructs added to OpenMP. The other author, i.e., the author that didn't present the introduction, will present the conclusion.** 

There is a way to enable locality-aware tasking in OpenMP 5.0 through the use of the clause `affinity` and through using parameters and hints to the clause. We need to facilitate for or create a larger number of task scheduling strategies taking into account in OpenMP. We propose having new types of hints for locality-aware task scheduling's clause `affinity` that specify (a) ~_what_~ data should be associated with a particular thread, or privatized and (b) ~_the degree_~ to which that data should be privatized. We believe that such support in task scheduling in OpenMP will further improve performance of OpenMP application codes and improve performance for a much larger number of application-architecture pairs. We would like to discuss the ideas with experts in OpenMP at the conference. We'll take feedback to add the ideas to OpenMP version 5.1 or a version of OpenMP immediately succeeding OpenMP version 5.1.