%% title: Enhancing Support in OpenMP to Improve Data Locality in Application Programs Using Task Scheduling 
%% author: Vivek Kale and Martin Kong. 
%% date: July 8th, 2018

\section{Data Locality for OpenMP Tasking}

\begin{frame}[label=motivationdlts]{Motivation}
\begin{itemize}
\item  In an OpenMP application in which work is scheduled to threads dynamically, \underline{data locality} is important for efficient execution of the application.
\item Using the clause {\tt affinity} for task scheduling proposed for OpenMP 5.0 can improve data locality~\cite{ompaffclause}.
\item However, strategies for tasking are fixed by OpenMP's runtime system, even with hints to the affinity clause.
\item One can argue that this small set of strategies isn’t beneficial for all application-architecture pairs~\cite{worksteal99, Kulkarni08schedulingstrategies}. \nocite{Olivier:2012:CMW:2388996.2389085}
\end{itemize} 
\end{frame}

\begin{frame}[label=posssoldlts]{A Possible Solution}
\begin{itemize}
\item OpenMP needs an adequate amount of support to maintain high levels of data locality when scheduling tasks to threads.
\item Specifically, we need task-to-thread affinity in OpenMP to reduce
\begin{enumerate} 
\item capacity cache misses on a multi-core node, or \textit{locality-awareness}, and 
\item coherence cache misses on a multi-core node, or \textit{locality-sensitivity}.
\end{enumerate}
\item We need to provide more hints to OpenMP's runtime for assigning OpenMP's tasks to threads in a way that preserves data locality.
\end{itemize}

\end{frame}

\begin{frame}[label=contributiondlts]{Contribution} 
\begin{itemize} 
 \item Our solution builds on the {\tt affinity} clause for OpenMP 5.0~\cite{OpenMP} $\rightarrow$ the user provides input to the clause as hints on
\begin{enumerate} 
\small \item \small \textit{what} data needs to be localized
\item \small the \textit{degree} to which the data should be localized
\end{enumerate}
\item Prior work on the degree to which the data should be localized has been shown to improve performance~\cite{dynwork}.
\item \underline{\textit{Contribution}}: the addition of constructs to OpenMP that provides and allow for a rich set of task scheduling schemes having (a) locality-awareness or (b) locality-sensitivity.
\item This work develops ideas of (a) for the affinity clause, and building on (b) from previous work for the affinity clause.
\end{itemize}
\end{frame}

\section{Data Scheduling}

\begin{frame}{Scheduling Data Access}
\begin{itemize}
    \item OpenMP lacks a mechanism for allowing the thread identifier to affect the scheduling of inner loops (when this is legal)
    \item Here we show two examples of how such mechanism can be used
    \item Benefits: Improve execution time, energy consumption and make better usage of available bandwidth
    \item We show the results of some preliminary experiments conducted to show the benefits of the proposed directive
\end{itemize}
\end{frame}

\begin{frame}[label=propdltsds]{Proposal}
\footnotesize{
\begin{itemize}
\item  Add {\bf loopshift} directive
\item Must be nested within a work-sharing directive and parallel region
\item Allow to map iterator of some inner loop of the work-sharing
      loop with some arithmetic expression
\item Can use pre-defined variables such as thread identifier (tid)
 and number of threads (numthreads)
\end{itemize}
}
```C, caption=OpenMP LoopShift Directive
#pragma omp parallel for 
for (i = lbi; i < ubi; i++)
{
  int j;
  pragma omp loopshift(j = (i + tid) % numthreads)
  for (j = lbj; j < ubj; j++)
  {
    /* do work */
  }
}
```

\end{frame}

\begin{frame}[label=exmmls]{Example: Matrix-Multiply Loop Shift}
\tiny{
\begin{itemize}
    \item First example: Matrix-Multiply
    \item Shift loop-K w.r.t outer parallel and worksharing loop-i
    \item Effect: Each thread accesses a different part of array B
    \item Example shows the semantics of {\bf loopshift} in terms of a more explicit worksharing loop
    \item Perform explicit partition of rows of B according to value of cc (core/thread)
    \item Could potentially use a renaming mechanism 
\end{itemize}
}
~~
\begin{minipage}{0.43\textwidth}
```C, caption=\footnotesize{MatMul Loop Shift Semantics}
#pragma omp parallel
{
 #pragma omp for private (i,j,kk)
 for (cc = 0; cc < CORES; cc++) {
  int lb = (ni/CORES) * cc;
  int ub = (ni/CORES) * (cc + 1);
  for (i = lb; i < ub; i++)
   for (kk = 0; kk < nk; kk++) {
    int k = (kk + cc*ni/CORES) % nk;
    for (j = 0; j < nj; j++)
      C[i][j] += A[i][k]  * B[k][j];
   }
 }
}
```
\end{minipage}
\hspace{0.5cm}
\begin{minipage}{0.48\textwidth}
```C, caption=\footnotesize{MatMul LoopShift Directive}
#pragma omp parallel
{
 #pragma omp for private (i,j,kk)
 for (cc = 0; cc < CORES; cc++) {
  int lb = (ni/CORES) * cc;
  int ub = (ni/CORES) * (cc + 1);
  for (i = lb; i < ub; i++)
   for (kk = 0; kk < nk; kk++) {
    #pragma omp loopshift \
      (k = (kk+cc*ni/CORES)%nk)
    for (j = 0; j < nj; j++)
     C[i][j] += A[i][k]  * B[k][j];
  }
 }
}
```
\end{minipage}
\end{frame}


\begin{frame}[label=exj2d]{Example: Jacobi 2D Stencil Loop Shift}
\footnotesize{
\begin{itemize}
    \item Second example: Jacobi-2D
    \item Shift loop-i w.r.t to thread number
    \item Effect: Ideally, user-provided mapping function should attempt to reuse the data already brought into cache by the thread
\end{itemize}
}
\begin{minipage}{0.43\textwidth}
```C, caption=\footnotesize{Jacobi Stencil LoopShift Semantics}
for (t = 0; t < TSTEPS; t++) {
 #pragma omp parallel
 {
  int ii;
  #pragma omp for private (j)
  for (ii = 1; ii < n-1; ii++) {
   int tid = omp_get_thread_num ();
   int i = (tid + ii) % (n-2) + 1;
   for (j = 1; j < n-1; j++)
    ref(B,i,j) = 0.2 * (
     ref(A,i,j) + ref(A,i-1,j) +
     ref(A,i+1,j) + ref(A,i,j-1) + 
     ref(A,i,j+1));
  }
 }
 /* pointer swap */
 temp = B; B = A; A = temp;
}
```
\end{minipage}
\hspace{0.5cm}
\begin{minipage}{0.48\textwidth}
```C, caption=\footnotesize{Jacobi Stencil LoopShift Directive}
for (t = 0; t < TSTEPS; t++) {
 #pragma omp parallel
 {
  // Assume tid is an ICV
  #pragma omp for private (j) \ 
    loopshift(i=(tid+ii)%(n-2)+1)
  for (int ii = 1; ii < n-1; ii++) {
   for (j = 1; j < n-1; j++)
    ref(B,i,j) = 0.2 * (
     ref(A,i,j) + ref(A,i-1,j) +
     ref(A,i+1,j) + ref(A,i,j-1) + 
     ref(A,i,j+1));
  }
 }
 /* pointer swap */
 temp = B; B = A; A = temp;
}
```
\end{minipage}
\end{frame}

\begin{frame}[label=prexps]{Preliminary Experiments}
\footnotesize{
\begin{itemize}
\item Performed some preliminary experiments on Intel Core i9-7900X  (10 core)
\item Used Clang v7.0 (llvm/trunk)
\item Experiments show that the {\bf loopshift} directive can be used to reduce execution time, improve bandwidth usage and/or reduce energy consumption 
\item We evaluate the kernels previously shown (matmul and jacobi-stencil 2D)
\item problem sizes ($750^2$ and $1000^3$)
\item the stencil iterates for 200 steps, we repeat the matmul kernel 10 times
\item Baseline versions assume static schedule
\end{itemize}
}
\end{frame}


\begin{frame}[label=resextime]{Results: Execution Time}
\footnotesize{
\begin{itemize}
\item Impact on execution time varies, in some cases we observe speedups, and in others the runtime remains constant
\item We didn't observe slowdowns
\item Need to perform a few more experiments
\end{itemize}
}
\begin{figure}
    \centering
    \includegraphics[scale=0.5]{figures/exectime.pdf}
\end{figure}
\end{frame}

\begin{frame}[label=resbwdv]{Results: Bandwidth and Data Volume}
\tiny{
\begin{itemize}
\item New directive allows multiple threads to better exploit bandwidth
\item Each threads can access disjoint memory regions
\item Common to observe higher data-volume movement, but observe no loss in performance
\item BW usage remains almost constant
\item Loopshift directive allows to affect memory traffic between L3 and DRAM
\item Have not observed effects between L1 and L2, nor L2 and L3
\item Require more experiments 
\end{itemize}
}
\begin{figure}
    \centering
    \includegraphics[scale=0.4]{figures/bw.pdf}
    \includegraphics[scale=0.4]{figures/datavolume.pdf}
\end{figure}
\end{frame}


\begin{frame}[label=resec]{Results: Energy Consumption}
\footnotesize{
\begin{itemize}
    \item Loopshift directive allows to reduce the energy consumption
    \item Small exploration shows energy reductions from 10\% to almost 70\%
    \item Does not compromise performance (from previous slides)
\end{itemize}
}
\begin{figure}
    \centering
    \includegraphics[scale=0.42]{figures/energy.pdf}
    \includegraphics[scale=0.42]{figures/energyred.pdf}
\end{figure}
\end{frame}

\begin{frame}[label=an]{Additional Notes}
\footnotesize{
\begin{itemize}
    \item New directive can alleviate different performance factors such as: execution time, bandwidth usage, memory traffic and energy consumption
    \item Directive, depending on application and code, can help emulate GPU SIMT access. 
    \item Also performed experiments with Pthreads: observed same behavior. 
    \item Also tested OpenMP with GCC 7.2, Clang runtime much faster in many cases
    \item GCC's OpenMP showed to be less sensitive to loopshifting (likely due to 
      some under-the-hood implementation)
\end{itemize}
}
\end{frame}

%\begin{frame}
%\frametitle{Data Scheduling}
%\begin{figure}[ht!]
%\includegraphics[width=0.9\textwidth]{figures/dgemm-smaller.pdf}
%\end{figure}
%\end{frame}


 %   \begin{frame}{Motivational Results (2)}
 %   \begin{figure}[ht!]
 %   \includegraphics[width=0.9\textwidth]{figures/dgemm-larger.pdf}
 %   \end{figure}
 %   \end{frame}
 %
 %   \begin{frame}{Motivational Results (3)}
 %   \begin{figure}[ht!]
 %   \includegraphics[width=0.9\textwidth]{figures/jacobi-smaller.pdf}
 %   \end{figure}
 %   \end{frame}
 %   
 %  \begin{frame}{Motivational Results (4)}
 %  \begin{figure}[ht!]
 %  \includegraphics[width=0.9\textwidth]{figures/jacobi-larger.pdf}
 %  \end{figure}
 %  \end{frame}

 \section{Locality-sensitivity in OpenMP}

\begin{frame}{Need to Use Task-to-thread Assignment History} 

- \small Consider an OpenMP application code  using a ```taskloop``` construct with multiple outer iterations and that computation load balanced across cores in a timestep. 
- If task scheduled to core different than the one in the previous outer iteration,  application code retrieves data from cache of the other core, causing a coherence cache miss $\rightarrow$ note that cost is high with more cores such as Intel Xeon Phi 64-core node.
- Such performance degradation is non-trivial if the \textbf{cost} of \textit{moving the data between the two cores} exceeds the \textbf{benefit} of \textit{load balancing obtained from migrating the data to another core}. 
%- Cost of coherence cache miss is high with more cores such as Intel Xeon Phi 64-core node. 
- Application programmer can improve data locality much more than just using the affinity clause may reduce such a cache miss in this case through hints to OpenMP runtime about how task affinity should be done.

\end{frame} 

\begin{frame}{Solution: Locality-sensitive Task Scheduling}

- \small \underline{\bf Key idea to improve locality-sensitivity}: specify a new tasking strategy that can be tuned so as to minimize synchronization and scheduling overheads. 
- Description and experimentation for one such scheduling strategy is shown in previous work @dynwork6. 
- Runtime system makes data of tasks be reused on a core as much as possible before that data must migrate to another core.
- Runtime system receives hints from the user on: 
    1. how to determine, based on history, which task a thread should execute next. 
    2. the number of queues to use in order to reduce synchronization costs and
    3. the likelihood that the data locality constraint will be ignored and a random task will be stolen from the queue. 
        
\end{frame} 

\begin{frame}{Proposal for Locality-sensitivity}

We propose adding a new scheduling strategy clause, ```schedstrat```, in which one uses the following parameters within the clause to specify how task scheduling ought to be done: 

 - `history(tid, mode)`: Specifies the mode, or methodology (from a pre-specified set of methodologies) in which history is used to select a task from the shared queue, given a thread ID. If no mode is chosen, the task is chosen based on whether it ran on a given thread ID in the previous outer iteration. 
 - `randomizationFactor`: Reduces coherence cache misses by having an adjustable parameter for the probability, between 0.0 and 1.0, that a task is chosen according to history from the previous outer iteration.

\end{frame}

% TODO: explain the code better. 

\begin{frame}{Proposal for Handling Locality-sensitivity}{Example illustrating Approach}

- \small Consider the Barnes-Hut code below run on a node of a supercomputer of four cores.

```C, caption=Barnes-Hut user code using proposed locality-sensitive tasking
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

\end{frame}

\begin{frame}{Support by Runtime}

- When each of the the four threads each pick up work from the shared work queue, a thread first generates a random number between  0.0 $<=$ P $<=$ 1.0. A user sets a threshold r. 
-  1. If p $>$ r,  dequeue a task that ran on thread X. 
   2. If p $<=$ r, choose a random thread that's not X and dequeue a task from that thread.

- If the number generated determines (1), the thread searches for the first task in the queue which has run on that thread in the previous invocation of the taskloop computation region. 
- If the thread finds such a task, the thread dequeues and the executes the task. If the thread doesn’t find such a task, the thread will dequeue the task at the head of the queue.

$\rightarrow$ Options passed to the affinity scheduling clause tunes the degree to which load balancing is done with respect to data locality.
\end{frame}

\begin{frame}{Implementation Guidelines}

The implementation: \\ 

- needs to not create false sharing in misalignment of shared queue; 
- should minimize time to search for a task in queue that match the locality tag;
- should reduce synchronization overheads by supporting a and tuning of parameter for number of queues;
- should use an efficient implementation of work stealing\cite{worksteal99};
- shall ideally have an automatic determination of parameters of the task scheduling strategy;
- should support history from previous outer iterations for per-outer-iteration adjustment of parameters of task scheduling strategy during runtime. 

\end{frame} 

\begin{frame}[label=perfExp]{Performance Expectations}

- There will be fewer coherence cache misses and less capacity cache misses with more memory bandwidth on the bus.
- Some benefits not addressed here but that can be addressed are:    

    1. The idea won’t decrease synchronization overheads.
    2. The prefetching engine still can’t be beneficial for constrained dynamic task scheduling because of the randomized branch involved in the strategy.
     
\end{frame}

\section{Closing}

\begin{frame}[label=dltsconclusions]{Conclusions}

1. \small Need mechanism to enable locality-aware and data-oriented task and thread scheduling in OpenMP 5.0 

2. \small Propose clause ```affinity``` and through using parameters and hints to the clause; propose ```loopshift```
directive to affect inner worksharing loops

3. Propose new types of  hints  for  locality-aware  task  scheduling’s  clause ```affinity``` that  specify
    - \footnotesize what data  should  be  associated  with  a  particular  thread,  or  privatized  
    - \footnotesize the degree to which that data should be privatized. 
    
4. We believe that such support in OpenMP  will improve  performance of many OpenMP application codes on current and future architectures. 
5. We’ll take feedback to add the ideas to OpenMP version 5.1 or a version of OpenMP immediately succeeding OpenMP version 5.1.

6. Please email us questions and inquiries :-)

\end{frame}


\begin{frame}[label=acksdlts]{Acknowledgements}

- This work was supported in part by funding Exascale Computing Project under Grant Number 17-SC-20-SC. 
- We thank input from Oscar Hernandez from Oak Ridge National Laboratory for initial input of the ideas for locality in tasking and formulating the ideas. 

\end{frame} 

\begin{frame}[c]{}

\centering {\Large Questions?}

\end{frame}


\begin{frame}[c]{Contacting Authors}

\begin{itemize} 
    \item Vivek Kale: \url{vkale\@isi.edu}
    \item Martin Kong: \url{mkong\@bnl.gov}
    \item Lingda Li: \url{lli\@bnl.gov}
\end{itemize} 

\end{frame}

%\begin{frame}[label=dltsproposal]{Proposal  for Data Locality for OpenMP}

%1. 
%2. 
%3.
%4. 

%\end{frame}


\begin{frame}[allowframebreaks]\renewcommand{\bibfont}{\footnotesize}
\frametitle{Bibliography}

\bibliographystyle{abbrvnat}
\bibliography{./mybibFile}

\end{frame}





%% --- Attic ----


%\begin{frame}{Need to Address the Lack of Locality-sensitivity } 

%- \small The non-trivial degradation can be avoided in OpenMP application code using tasking if the tasks are assigned to threads __carefully__ so as to avoid the coherence cache miss. 
%- True in particular when the load imbalance of the computation of the application %code exhibits a repeated pattern over application timesteps, i.e., is %__persistent__@CharmppOOPSLA93, and the data that a core uses across outer iterations %has a complementary pattern.

%- True in particular when the load imbalance of the computation of the application %code exhibits a repeated pattern over application timesteps, i.e., is %__persistent__@CharmppOOPSLA93, and the data that a core uses across outer iterations %has a complementary pattern.
%\end{frame}

%\lstinputlisting{listingsDir/sampleStencil.c} 

%}