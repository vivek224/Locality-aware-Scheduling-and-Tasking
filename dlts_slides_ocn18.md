%% title: Enhancing Support in OpenMP to Improve Data Locality in Application Programs Using Task Scheduling 
%% author: Vivek Kale and Martin Kong. 
%% date: July 8th, 2018

\begin{frame}{Motivation}{Problem}

- Data locality is important for efficient execution of an OpenMP application in which work is assigned or scheduled to threads dynamically.
- Using the clause ‘affinity’ for task scheduling proposed for OpenMP 5.0 can improve data locality [@ompaffclause; @dynwork2].
- However, the task scheduling strategies are fixed by the runtime system, even with the hints available to the affinity to the affinity clause.
- While having a few fixed task scheduling scheduling strategies taking into account data locality can be beneficial for most applications-architecture pairs, it is arguable that this small set of task scheduling strategies isn’t beneficial for all applications-architecture pairs \cite{dynwork2, dynwork, worksteal99, DonfackMulticore, DPLASMA, Kulkarni08schedulingstrategies}. \nocite{Olivier:2012:CMW:2388996.2389085}


\end{frame}

\begin{frame}{Motivation}{Fix}

- \small OpenMP needs adequate amount of support to maintain high levels of data locality when scheduling tasks to threads. 
- We need to provide a larger numberof hints to the OpenMP runtime of how to assign OpenMP tasks to threads in a way that preserves data locality.
- Specifically, we need 
   1. task-to-thread affinity in OpenMP to reduce capacity cache misses on a multi-core node, which we’ll refer to as locality-awareness, and 
   2. task-to-thread affinity in OpenMP to reduce coherence cache misses on a multi-core node, which we’ll refer to as locality-sensitivity. 
- Builds on the affinity clause for OpenMP 5.0[@OpenMP] $\rightarrow$ the user provides input to the clause as hints on \textit{what} data needs to be localized and the \textit{degree} to which the data should be localized. 
- Contribution: the addition of constructs to OpenMP that provides and allow for a rich set of task scheduling schemes having locality-awareness or locality-sensitivity. We focus on developing (a) and building on (b) from previous work.

\end{frame} 

\begin{frame}{Conclusion}

\end{frame} 

\begin{frame}[plain]
\renewcommand{\bibfont}{\footnotesize}
\frametitle{Bibliography}

\bibliographystyle{abbrvnat}
\bibliography{mybibFile}

\end{frame}