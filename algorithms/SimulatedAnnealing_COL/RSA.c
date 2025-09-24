/* by FRT */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)
#define sign(x) ((x) > 0 ? 1 : -1)
#define max(a,b) ((a) > (b) ? (a) : (b))

#define Q 5                                           //Q = number of colors

struct vector {
  int n[Q];
} zero, sum, group[Q], *perm, *sumColors, maxMagn[Q];   //

int N, NoverQ, M, R, *graph, **neigh, **neigh_pos, *deg, **color, fact[Q+1];

/* variabili globali per il generatore random */
unsigned myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

//random init of random number generator
unsigned randForInit(void) {
  unsigned long long y;
  
  y = myrand * 16807LL;
  myrand = (y & 0x7fffffff) + (y >> 31);
  if (myrand & 0x80000000) {
    myrand = (myrand & 0x7fffffff) + 1;
  }
  return myrand;
}

//same
void initRandom(void) {
  int i;
  ip = 128;    
  ip1 = ip - 24;    
  ip2 = ip - 55;    
  ip3 = ip - 61;
  
  for (i = ip3; i < ip; i++) {
    ira[i] = randForInit();
  }
}

//Box-Muller to have a Gaussian distributed random number
float gaussRan(void) {
  static int iset = 0;
  static float gset;
  float fac, rsq, v1, v2;
  
  if (iset == 0) {
    do {
      v1 = 2.0 * FRANDOM - 1.0;
      v2 = 2.0 * FRANDOM - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return v2 * fac;
  } else {
    iset = 0;
    return gset;
  }
}

void error(char *string) {
  fprintf(stderr, "ERROR: %s\n", string);
  exit(EXIT_FAILURE);
}

//create all vectors and init them with all zeros
void allocateMem(void) {
  int i;
  
  fact[0] = 1;
  for (i = 0; i < Q; i++) {
    zero.n[i] = 0;
    fact[i+1] = (i+1) * fact[i];
  }
  deg = (int*)calloc(N, sizeof(int));
  neigh = (int**)calloc(N, sizeof(int*));
  neigh_pos = (int**)calloc(N, sizeof(int*));
  color = (int**)calloc(R, sizeof(int*));
  for (i = 0; i < R; i++)
    color[i] = (int*)calloc(N, sizeof(int));
  perm = (struct vector*)calloc(fact[Q], sizeof(struct vector));
  sumColors = (struct vector*)calloc(N, sizeof(struct vector));
}

//same as colBP
void printPerm(void) {
  int i, j;
  
  for (i = 0; i < fact[Q]; i++) {
    for (j = 0; j < Q; j++)
      printf("%i", perm[i].n[j]);
    printf("\n");
  }
  printf("\n");
}

//same as colBP
struct vector rotate(struct vector input, int modulo) {
  int i;
  struct vector output;

  for (i = 0; i < modulo; i++)
    output.n[i] = (input.n[i] + 1) % modulo;
  return output;
}

//same as colBP
void initPerm(int max) {
  int i;
  
  if (max == 1)
    perm[0].n[0] = 0;
  else {
    initPerm(max-1);
    for (i = 0; i < fact[max-1]; i++)
      perm[i].n[max-1] = max-1;
    for (; i < fact[max]; i++)
      perm[i] = rotate(perm[i-fact[max-1]], max);
  }
}

//same as colBP
void makeGraph(void) {
  int i, var1, var2;

  for (i = 0; i < N; i++)
    deg[i] = 0;
  for (i = 0; i < M; i++) {
    var1 = (int)(FRANDOM * N);
    do {
      var2 = (int)(FRANDOM * N);
    } while ((int)(var1/NoverQ) == (int)(var2/NoverQ));
    graph[2*i] = var1;
    graph[2*i+1] = var2;
    deg[var1]++;
    deg[var2]++;
  }
  for (i = 0; i < N; i++) {
    neigh[i] = (int*)calloc(deg[i], sizeof(int));
    deg[i] = 0;
  }
  for (i = 0; i < M; i++) {
    var1 = graph[2*i];
    var2 = graph[2*i+1];
    neigh[var1][deg[var1]++] = var2;
    neigh[var2][deg[var2]++] = var1;
  }
}

void makeGraph_noPlanted(void) {
  int i, var1, var2;
  for (i = 0; i < N; i++)
    deg[i] = 0;                         //init all degrees of nodes at 0
  for (i = 0; i < M; i++) {
    var1 = (int)(FRANDOM * N);                          //connect node var1 with var2
    do {
      var2 = (int)(FRANDOM * N);                       
    } while (var1 == var2); //var1/NoverQ has to be different from vr2/noverQ: always different colors
    graph[2*i] = var1;                                  //in this way var1 and var2 are linked
    graph[2*i+1] = var2;                                // the same
    deg[var1]++;                                        //increase the degrees
    deg[var2]++;
  }
  for (i = 0; i < N; i++) {
    neigh[i] = (int*)calloc(deg[i], sizeof(int));       //every element of array neigh is an array of length deg[i]
    neigh_pos[i] = (int*)calloc(deg[i], sizeof(int));
    deg[i] = 0;
  }
  for (i = 0; i < M; i++) {                             //smart way to place var2 in the neighborhood of var1, and viceversa
    var1 = graph[2*i];
    var2 = graph[2*i+1];
    neigh[var1][deg[var1]] = var2;                      
    neigh_pos[var1][deg[var1]] = deg[var2];

    neigh[var2][deg[var2]] = var1;
    neigh_pos[var2][deg[var2]] = deg[var1];

    deg[var1]++;
    deg[var2]++;
  }
}

void Alloc(){
  int i;
  //neigh=(int **)malloc(N*sizeof(int*));
  for(i=0;i<N;i++){
    neigh[i]=(int *)calloc(deg[i],sizeof(int));
    neigh_pos[i] = (int*)calloc(deg[i], sizeof(int));
    deg[i] = 0;
  }
}

void create_A(FILE *f){
  int i,j,n;
  fscanf(f,"N   %d\n",&N);
  fscanf(f,"M   %d\n",&M);
  
  for(n=0;n<N;n++){
    deg[n] = 0;
  }
  for(n=0;n<M;n++){
    fscanf(f,"e   %d   %d\n",&i,&j);
    i=i-1;
    j=j-1;
    graph[2*n] = i;                                  //in this way var1 and var2 are linked
    graph[2*n+1] = j;  
    deg[i]++;
    deg[j]++;
  }
  rewind(f);
  Alloc();
  fscanf(f,"N   %d\n",&N);
  fscanf(f,"M   %d\n",&M);
  for(n=0;n<M;n++){
    fscanf(f,"e   %d   %d\n",&i,&j);
    i=i-1;
    j=j-1;
    neigh[i][deg[i]]=j;
    neigh_pos[i][deg[i]] = deg[j];
    neigh[j][deg[j]]=i;
    neigh_pos[j][deg[j]] = deg[i];
    deg[i]++;
    deg[j]++;
  }
  fclose(f);
}


//??
void initColors(void) {
  int i, j;
  
  for (i = 0; i < N; i++)
    sumColors[i] = zero;
  for (i = 0; i < R; i++)                  //What is R?
    for (j = 0; j < N; j++) {
      color[i][j] = (int)(FRANDOM * Q);    //set a random color in 
      sumColors[j].n[color[i][j]]++;
    }
}

//one montecarlo step
void oneMCS(int * wc, double temperature, double gamma) {
  int i, j, newColor;
  double deltaUnsat;

  for (i = 0; i < N; i++) {
    do {
      newColor = (int)(FRANDOM * Q);                      //pick a new color different from the previous one
    } while (newColor == wc[i]);
    sum = zero;
    for (j = 0; j < deg[i]; j++)
      sum.n[wc[neigh[i][j]]]++;
      deltaUnsat = sum.n[newColor] - sum.n[wc[i]] + gamma * (sumColors[i].n[wc[i]] - 1 - sumColors[i].n[newColor]);
    if (deltaUnsat <= 0 || deltaUnsat < -temperature * log(FRANDOM)) {
      sumColors[i].n[wc[i]]--;
      sumColors[i].n[newColor]++;
      wc[i] = newColor;
    }
  }
}

int computeUnsat(int * wc) {
  int i, res = 0;

  for (i = 0; i < M; i++)
    res += (wc[graph[2*i]] == wc[graph[2*i+1]]);
  return res;
}

double overlapPlanted(int * wc) {
  int i, j, overlap, maxOver=0;

  for (i = 0; i < Q; i++)
    group[i] = zero;
  for (i = 0; i < N; i++)
    group[(int)(i/NoverQ)].n[wc[i]]++;
  for (i = 0; i < fact[Q]; i++) {
    overlap = 0;
    for (j = 0; j < Q; j++)
      overlap += group[j].n[perm[i].n[j]];
    if (overlap > maxOver) maxOver = overlap;
  }
  return (double)(Q*maxOver-N)/(Q-1)/N;
}

double overlap(int * wc1, int * wc2) {
  int i, res = 0;

  for (i = 0; i < N; i++)
    res += (wc1[i] == wc2[i]);
  return (double)(Q*res-N)/N/(Q-1);
}

void printReplicaOverlapMoments(void) {
  int i1, i2, cont = 0;
  double tmp, mean = 0.0, var = 0.0;

  for (i1 = 0; i1 < R-1; i1++)
    for (i2 = i1 + 1; i2 < R; i2++) {
      tmp = overlap(color[i1], color[i2]);
      mean += tmp;
      var += tmp * tmp;
      cont++;
    }
  if (cont) {
    mean /= cont;
    var /= cont;
    printf(" %g %g", mean, sqrt(var-mean*mean));
  }
}

void printMaxMagn(int *col){
  int i=0,j,k,max;

  for(j=0;j<Q;j++){
    for(k=0;k<Q;k++){
      maxMagn[j].n[k]=0;
    }
    for(;i<(N/Q)*(j+1);i++){
      maxMagn[j].n[col[i]]++;
    }
  }
  for(j=0;j<Q;j++){
    max=0;
    for(k=0;k<Q;k++){
      if(maxMagn[k].n[j]>max)max=maxMagn[k].n[j];
    }
    printf(" %g ",(double)(max*Q)/N-1./Q);
  }
  //  printf("\n");
}

void freeMem(void) {
  int i;
  
  for (i = 0; i < N; i++)
    free(neigh[i]);
    free(neigh_pos[i]);
}

//start always in the paramagnetic phase
//R=1, gamma=0
int main(int argc, char *argv[]) {
  int i, j, nIter, is, nSamples, measTime, energy;
  double c, c_min, c_max, delta_c, temp, startTemp, gamma;
  FILE *f;
  char file_name[256];

  if (argc != 11 && argc!=12) {
    fprintf(stderr, "usage: %s <N> <c_min> <c_max> <delta_c> <startTemp> <nIter> <nSamples> <R> <gamma> [seed]\n", argv[0]);
    exit(EXIT_FAILURE);
  } //R number of replicas

  if (argc ==12) {
    myrand = (unsigned)atoi(argv[11]);
  }else{
    FILE *devran = fopen("/dev/urandom","r");
    fread(&myrand, 4, 1, devran);
    fclose(devran);
  }

  N = atoi(argv[1]);
  c_min = atof(argv[2]);
  c_max = atof(argv[3]);
  delta_c = atof(argv[4]);
  startTemp = atof(argv[5]);
  nIter = atoi(argv[6]);
  nSamples = atoi(argv[7]);
  R = atoi(argv[8]);
  gamma = atof(argv[9]);
  char *baseName = argv[10];

  //if (Q * (int)(N/Q) != N){
  //  error("Q must divide N");
  //} 
  //NoverQ = (int)(N/Q);
  //printf("# Q = %i   N = %i   M = %i   c = %f   startTemp = %f   nIter = %i   seed = %u\n",
	//Q, N, M, c, startTemp, nIter, myrand);
  initRandom();
  allocateMem();
  printf("# N M id E nIter\n");
  for (is = 0; is < nSamples; is++) {
      //makeGraph();
      
    for(c=c_min; c<=c_max; c+= delta_c){
      M = (int)(0.5 * c * N + 0.5);
      graph = (int*)realloc(graph, 2*M*sizeof(int));
      measTime = (int)(nIter / 100);
      gamma /= R;
      initPerm(Q);
      //printPerm();
      file_name[0] = "\0";
      snprintf(file_name, sizeof(file_name), "%sErdosRenyi_N_%d_M_%d_id_%d.txt", baseName, N, M, is + 1);
      f = fopen(file_name,"r");
      create_A(f);
      initColors();
      temp = startTemp;
      for (i = nIter; i > 0; i--) {
        temp = i * startTemp / nIter;
        for (j = 0; j < R; j++){
	        oneMCS(color[j], temp, gamma);
          if (i % measTime == 0) {
              energy = computeUnsat(color[j]);
            if(energy==0){
                i = -1;
            }
          }
        }
        fflush(stdout);
      }
      for (j = 0; j < R; j++){
        printf("%d %d %d %d %d\n", N, M, is+1, computeUnsat(color[j]), nIter); 
      }
      freeMem();
    }
    //fprintf(stderr, "c= %f\n", c);
  }
  return EXIT_SUCCESS;
}
