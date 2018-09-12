while (timestep < 1000 )
{
 #pragma omp parallel for schedule(static)
    {
    for(i= 0; i<128; i++;)
    {
      lastThreadRunOn[i] = omp_get_thread_num(); 
      c[i] += a[i]*b[i]; 
      MPI_Allreduce();
    timestep++;
    }
  }
}