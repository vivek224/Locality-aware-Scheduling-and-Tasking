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