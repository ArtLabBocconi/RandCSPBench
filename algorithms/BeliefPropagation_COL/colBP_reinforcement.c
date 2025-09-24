/* by MCA & FD */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)
#define sign(x) ((x) > 0 ? 1 : -1)
#define max(a,b) ((a) > (b) ? (a) : (b))

#define Q 5
#define DAMP 0

//gcc colBP_reinforcement.c -std=c99 -lm -o colBP_reinforcement.exe

struct vector {
  int n[Q];
} zero, group[Q], *perm;

/*
- zero is a zero-struct
- group is an array of struct
*/

int N, NoverQ, M, *graph, **neigh, **neigh_pos, *deg,  fact[Q+1];
double c_out,p_out;

/*
- N is the number of nodes
- NoverQ is (int) N/Q
- graph is an array with all the edges: graph[0] and graph[1] have the start and the end of a link (undirected)
- neigh and neigh_pos are array of array (each array is the neighborhood of the )
- deg has the number of neighbors of each node
- fact??
*/

/* Global variables for random numbers generation */
unsigned myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

//Random generator
unsigned randForInit(void) {
  unsigned long long y;
  
  y = myrand * 16807LL;
  myrand = (y & 0x7fffffff) + (y >> 31);
  if (myrand & 0x80000000) {
    myrand = (myrand & 0x7fffffff) + 1;
  }
  return myrand;
}

//Random generator
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

//Random gaussian number with Box-Muller
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

//Print if there is an error
void error(char *string) {
  fprintf(stderr, "ERROR: %s\n", string);
  exit(EXIT_FAILURE);
}

//Dynamical arrays
void allocateMem(void) {
  int i;

  fact[0] = 1;
  for ( i = 0; i < Q; i++) {
    zero.n[i] = 0;
    fact[i+1] = (i+1) * fact[i];
  }
  deg = (int*)calloc(N, sizeof(int));
  neigh = (int**)calloc(N, sizeof(int*));
  neigh_pos = (int**)calloc(N, sizeof(int*));
  perm = (struct vector*)calloc(fact[Q], sizeof(struct vector));
}

//
void printPerm(void) {
  int i, j;
  
  for (i = 0; i < fact[Q]; i++) {
    for (j = 0; j < Q; j++)
      printf("%i", perm[i].n[j]);
    printf("\n");
  }
  printf("\n");
}

//non ho capito
struct vector rotate(struct vector input, int modulo) {
  struct vector output;
  int i;

  for (i = 0; i < modulo; i++)
    output.n[i] = (input.n[i] + 1) % modulo;
  return output;
}

//neanche qui
void initPerm(int max) {
  int i;
  
  if (max == 1)
    perm[0].n[0] = 0;
  else {
    initPerm(max-1);
    for (i = 0; i < fact[max-1]; i++)          //perm[i] is an array of permutation: perm[0]=[1,2,3], perm[1]=[2,1,3], etc...
      perm[i].n[max-1] = max-1;                //so perm[i].n is an array of length Q
    for (; i < fact[max]; i++)
      perm[i] = rotate(perm[i-fact[max-1]], max);
  }
}

//create a Erdos-Renyi random Graph: G(N,M), N nodes and M links
void makeGraph_planted(void) {
  int i, var1, var2;
  for (i = 0; i < N; i++)
    deg[i] = 0;                         //init all degrees of nodes at 0
  for (i = 0; i < M; i++) {
    var1 = (int)(FRANDOM * N);                          //connect node var1 with var2
    do {
      var2 = (int)(FRANDOM * N);                       
    } while ((int)(var1/NoverQ) == (int)(var2/NoverQ)); //var1/NoverQ has to be different from vr2/noverQ: always different colors
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

//Maybe compute the overlap with the planted solution
//Problem is: there is a symmetry between permutation of colors, so every permutation has to be checked
double overlapPlanted(int *color){
  int i, j, overlap, maxOver=0;                 //overlap computed, and max over all overlaps

  for (i = 0; i < Q; i++)
    group[i] = zero;                            //init the array of struct group with the struct zero in each position
  for (i = 0; i < N; i++)
    group[(int)(i/NoverQ)].n[color[i]]++;       //????
  for (i = 0; i < fact[Q]; i++) {
    overlap = 0;
    for (j = 0; j < Q; j++)
      overlap += group[j].n[perm[i].n[j]];
    if (overlap > maxOver) maxOver = overlap;
  }
  return (double)(Q*maxOver-N)/(Q-1)/N;
}

//
void freeMem(double ***theta) {
  int i,j;
  for (i = 0; i < N; i++){
    free(neigh[i]);
    free(neigh_pos[i]);
      for(j=0;j<deg[i];j++){
	free(theta[i][j]);
      }
      free(theta[i]);
    }

}

//init random cavity marginals
void init_theta(double **theta_i, double ***theta){     
  int i,j,q;
  double sum,ran;

  for(i=0;i<N;i++){
    sum=0;
    for(q=0;q<Q;q++){
      ran=FRANDOM;
      theta_i[i][q]=ran;                                //init random probabilities
      sum+=ran;
    }
    for(q=0;q<Q;q++){
      theta_i[i][q]/=sum;                               //to normalize probabilities over all q colors
    }
    for(j=0;j<deg[i];j++){
      for(q=0;q<Q;q++){
	      theta[i][j][q]=theta_i[i][q];
      }
    }
  }
}

//?? Not used
void init_theta_planted(double **theta_i, double ***theta){
  int i,j,q;
  double sum,ran;

  for(i=0;i<N;i++){
    sum=0;
    for(q=0;q<Q;q++){
      theta_i[i][q]=(q==(int)(i/NoverQ)?0.99:0.01);      //if q==(int) i/NoverQ, than theta_i[i][q]=0.99, else =0.01
      sum+=theta_i[i][q];
    }
    for(q=0;q<Q;q++){
      theta_i[i][q]/=sum;
    }
    for(j=0;j<deg[i];j++){
      for(q=0;q<Q;q++){
	      theta[i][j][q]=theta_i[i][q];
      }
    }
  }
}

//Update mu values
void compute_expmu(double ***theta, double **expmu, double gamma_t){
    int *n_ch, *pos;
    double temp;
    for(int i=0; i<N; i++){
        n_ch=neigh[i]; pos=neigh_pos[i];
        for(int q=0; q<Q; q++){
            temp = 1.;
            for(int k=0; k<deg[i]; k++){
                temp *= (1-theta[n_ch[k]][pos[k]][q]);
            }
            if(temp != 0){
                expmu[i][q] = pow(temp, (1-gamma_t));
            }else{
                expmu[i][q] = 0;
            }
        }
    }
}

//If all probs are equal to zero, entropic zero T limit
void find_min(int Nzero[Q], double *theta_new, int chosen, int j, double ***theta, double **expmu){
  double z, res;
  int i, q, n_min,i_min[Q],k, min=deg[chosen]+1;
  /*for(int q=0; q<Q; q++){
    i_min[q] = q;
  }*/
  //n_min = Q; //in this way, if all prohibited, extract a random color
  for(q=0;q<Q;q++){
    if(Nzero[q]<min){
      n_min=1;
      i_min[0]=q;
      min=Nzero[q];
    }else{
      if(Nzero[q]==min){
        if((n_min<0)||n_min>=Q){
          printf("n_min=%d q=%d min=%d\n", n_min, q, min);
        }
	      i_min[n_min]=q;
	      n_min++;
      }
    }
  }
  for(q=0;q<Q;q++){
    theta_new[q]=0.;
  }
  if(n_min==1){
    theta_new[i_min[0]]=1.;
  }else{
    z=0.;
    for(q=0;q<n_min;q++){
        if((n_min<0)||n_min>Q){
          printf("n_min=%d q=%d min=%d, i_min[q]=%d\n", n_min, q, min, i_min[q]);
        }
      theta_new[i_min[q]]=1.;
      for(k=0;k<deg[chosen];k++){
	      if(k!=j){
            res = expmu[chosen][i_min[q]]*(1.-theta[neigh[chosen][k]][neigh_pos[chosen][k]][i_min[q]]);
	        if(res!=0.0){
            // ATTENTION!! reinforcement used also in find min !!!
	          theta_new[i_min[q]]*= res; 
	        }
	      }
      }
      z+=theta_new[i_min[q]];
    }
    if(z!=0){
      for(i=0;i<n_min;i++){
        theta_new[i_min[i]]/=z;
      }  
    }else{
      for(i=0;i<n_min;i++){
        theta_new[i_min[i]]=1/((double)n_min);
      }  
    }  
  }

}

//Iterate till t_MAX or convergence the probabilities
int iteration(int t_MAX,double **theta_i, double ***theta, double **expmu, double gamma, double dt){
  int t,r,i,j,k,*n_ch,*pos,q,Nzero[Q],chosen_q;
  double diff=1,theta_new[Q],z, gamma_t, res, eps=1e-30;
  gamma_t = pow(gamma,1/dt);
  for(t=0;t<t_MAX&&diff>1e-7;t++){
    diff=0.; //f=0;
    compute_expmu(theta, expmu, gamma_t);
    for(r=0;r<N;r++){                                       //N updates
      i=(int)(FRANDOM*(double)N);                           //random node to update
      n_ch=neigh[i]; pos=neigh_pos[i];
      for(j=0;j<deg[i];j++){                                //update of theta[i][j]
	      z=0;
	      for(q=0;q<Q;q++){                                   //compute theta_{i->j}(s_i) and store it in theta_new[q]
	        theta_new[q]=1;                                   //that is the marginal of i excluding the node j
	        Nzero[q]=0;
	        for(k=0;k<deg[i];k++){                            //running over all neighbors k of i
	          if(k!=j){                                       //cavity marginal expression, at T=0:
              theta_new[q]*=expmu[i][q]*(1.-theta[n_ch[k]][pos[k]][q]); //entrambi divisi per d per avere un numero di ordine 1
	            if((1.-theta[n_ch[k]][pos[k]][q])==0.0){
		            Nzero[q]++;                                 //count the number of zeros chosing a certain color
                                                            //to make the entropic T=0 limit, useful iif 
	            }
	          }
	        }
	        z+=theta_new[q];                                  //z is the normalization; 
 	      }
	      if(z!=0.0){                                         //if different from zero, it is possible to normalize
	        for(q=0;q<Q;q++){
	          theta_new[q]/=z;                                                   //convex combination to damp the algorithm
	          theta_new[q]=DAMP*theta[i][j][q]+(1.-DAMP)*theta_new[q];           //DAMP is defined as 0.5
	          diff+=fabs((theta_new[q]-theta[i][j][q])/(theta_new[q]+eps));      //diff between old and new
	          theta[i][j][q]=theta_new[q];   
	        }
	      }else{ //if all probs are equal to zero, entropic zero T limit with function find_min
	        find_min(Nzero,theta_new,i,j,theta, expmu);                                 
	        for(q=0;q<Q;q++){
	          theta_new[q]=DAMP*theta[i][j][q]+(1.-DAMP)*theta_new[q];
	          diff+=fabs((theta_new[q]-theta[i][j][q])/(theta_new[q]+eps)); 
	          theta[i][j][q]=theta_new[q];
	        }
	      }         
      }
    }
    gamma_t *= gamma;
  }

  for(i=0;i<N;i++){
    n_ch=neigh[i]; pos=neigh_pos[i];
    z=0;
    for(q=0;q<Q;q++){ //aggiorno theta[i]
      theta_new[q]=1;
      Nzero[q]=0;
      for(k=0;k<deg[i];k++){
        // ATTENTION!! Also here used mu
	      res = theta_new[q]*=expmu[i][q]*(1.-theta[n_ch[k]][pos[k]][q]); 
	      if((res==0.0)){
	        Nzero[q]++;
	      }
      }
      z+=theta_new[q];
    }
    if(z!=0.0){
      for(q=0;q<Q;q++){
	      theta_new[q]/=z;
	      theta_new[q]=DAMP*theta_i[i][q]+(1.-DAMP)*theta_new[q];
	      theta_i[i][q]=theta_new[q];   
      }
    }else{
      find_min(Nzero,theta_new,i,-1,theta, expmu);
      for(q=0;q<Q;q++){
	      theta_new[q]=DAMP*theta_i[i][q]+(1.-DAMP)*theta_new[q];
	      theta_i[i][q]=theta_new[q];   
      }
    }
  }

  return t;
}

//check if any variable has all colors prohibited
int check_contradiction(int **prohib_cols, int N){
  int contradiction, zero_probs;
  contradiction = 0;
  for(int i=0; i<N; i++){
    zero_probs = 0;
    for(int q=0; q<Q; q++){
      if(prohib_cols[i][q]!=1){
        zero_probs = 0;
        break;
      }else{
        zero_probs ++;
      }
    }
    if(zero_probs == Q){                    //if all cols prohibited, contradiction=1
      contradiction += 1;
    }
  }
  return contradiction;
}

//Assign the color with the max prob in theta, being theta shape = [Q]
int assign_color(double *theta){
  double max=-1;
  int q,n_max,i_max[Q];               //i_max is the color with the max probability

  /*for(q=0;q<Q; q++){
    //i_max[q] = q;
    if(theta[q]<0 || theta[q]>=Q){
        printf("theta= %f\n", theta[q]);
    }
  }*/
  for(q=0;q<Q;q++){
    if(theta[q]>max){
      n_max=1;
      i_max[0]=q;
      max=theta[q];
    }else{
      if(theta[q]==max){
	      i_max[n_max]=q;                 //if two or more colors have the same prob, save all, later will pick a random color among them
	      n_max++;
      }
    }
  }
  if((i_max[0]>Q) || (i_max[0]<0)){
    printf("%d %d %f\n", i_max[0], n_max, max);
    printf("%f %f %f %f %f\n", theta[0], theta[1], theta[2], theta[3], theta[4]);
  }
  if(n_max==1){
    return i_max[0];
  }else{
    return i_max[(int)(FRANDOM*n_max)]; //if there are two or more colors pick a random one
  }
}

//m2 is the second moment of the probabilities theta_i[i][j]
void printMoments(double **theta_i){

  int i,j;
  double m2=0;

  for(i=0;i<N;i++){
    for(j=0;j<Q;j++){
      m2+=pow(theta_i[i][j],2);
    }
  }
  printf("%g\n",m2/(double)N);
}

//N number of nodes, c mean connectivity, nIter max number of iterations for BP to converge, nSamples number of random graphs to create
int main(int argc, char *argv[]) {                     //params: N, c, nIter, nSamples (eventually: myrand)
  int i, j, nIter, is, nSamples, t, connected_node, contradiction, *color, **prohib_cols;
  double c, c_min, c_max, delta_c, gamma, dt, ***theta,**theta_i, **expmu;
  FILE *f;
  char file_name[256];

  if (argc!=11 && argc!=10) {
    fprintf(stderr, "usage: %s <N> <c_min> <c_max> <delta_c> <nIter> <nSamples> <gamma> <dt> [seed]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  N = atoi(argv[1]);
  c_min = atof(argv[2]);
  c_max = atof(argv[3]);
  delta_c = atof(argv[4]);
  nIter = atoi(argv[5]);
  nSamples = atoi(argv[6]);
  gamma = atof(argv[7]);
  dt = atof(argv[8]);
  char *baseName = argv[9];

  if (argc == 11) {                                     //random selcted seed
    myrand = (unsigned)atoi(argv[10]);
    if (myrand == 2147483647)
      error("seed must be less than 2147483647");
  } else {                                             //otherwise random seed
    FILE *devran = fopen("/dev/random","r");         
    fread(&myrand, 4, 1, devran);
    fclose(devran);
  }

  //if (Q * (int)(N/Q) != N) 
  //  error("Q must divide N");

  theta=(double ***)malloc(N*sizeof(double**));         //theta has a shape [N,deg[i],Q]
  theta_i=(double **)malloc(N*sizeof(double*));  
  expmu=(double **)malloc(N*sizeof(double*));       
  color=(int *)malloc(N*sizeof(int));                   //color has a shape [N]
  prohib_cols = (int **)malloc(N*sizeof(int*));
  for(i=0;i<N;i++){
    prohib_cols[i] = (int *)malloc(Q*sizeof(int));
  }

  for(i=0;i<N;i++){
    theta_i[i]=(double *)malloc(Q*sizeof(double));
    expmu[i] = (double *)malloc(Q*sizeof(double));
  }

  initRandom();
  allocateMem();
  initPerm(Q);

  //printf("# Q = %i   N = %i   M = %i   c = %f   nIter = %i  gamma = %f  dt = %f  seed = %u\n",
	//Q, N, M, c, nIter, gamma, dt, myrand);
  printf("# N M id E nIter gamma dt\n");

  for(c=c_min; c<=c_max; c+=delta_c){
    NoverQ = (int)(N/Q);
    M = (int)(0.5 * c * N + 0.5);                         //number of links M to have a mean connectivity c
    graph = (int*)realloc(graph, 2*M*sizeof(int));
    //#####################################                 ALGORITHM
    for (is = 0; is < nSamples; is++) {
      //makeGraph_planted();                                //make graph
      //makeGraph_noPlanted();
      file_name[0] = "\0";
      snprintf(file_name, sizeof(file_name), "%sErdosRenyi_N_%d_M_%d_id_%d.txt", baseName, N, M, is + 1);
      f = fopen(file_name,"r");
      create_A(f);
        //theta has to be resized depending on new graph
      theta=(double ***)realloc(theta, N*sizeof(double**));         //theta has a shape [N,deg[i],Q]
      for(i=0;i<N;i++){                                     //define theta_i and theta
        theta[i]=(double **)malloc(deg[i]*sizeof(double*));
        for(j=0;j<deg[i];j++){
  	      theta[i][j]=(double *)malloc(Q*sizeof(double));
        }
      }
      for(int i=0; i<N; i++){
        for(int q=0; q<Q; q++){
          prohib_cols[i][q] = 0;
        }
      }
      init_theta(theta_i, theta);
      t=iteration(nIter, theta_i, theta, expmu, gamma, dt);                   //t of convergence or nIter if reached
      //After computing all probabilities theta:
      for(i=0;i<N;i++){
        color[i]=assign_color(theta_i[i]);                  //assign the color of every node 
        for(int j=0; j<deg[i]; j++){
          connected_node = neigh[i][j];
          prohib_cols[connected_node][color[i]] = 1;         //write 1 at prohibited color
        }
      }
      contradiction = check_contradiction(prohib_cols, N);
      printf("%d %d %d %d %d %f %f\n", N, M, is+1, contradiction, nIter, gamma, dt); 
      fflush(stdout);      
      freeMem(theta);
    }
  }
  //#######################################
  return EXIT_SUCCESS;
}
