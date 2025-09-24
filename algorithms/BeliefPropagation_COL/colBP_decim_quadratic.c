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

#define Q 3
#define DAMP 0

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

void quicksort(double *a, int *n_a, int size){
  int i,k=size-1,minsize=0,tempint;
  double temp,last;

  if(size==2){
    if (a[0]>a[1]){
      temp=a[1];
      a[1]=a[0];
      a[0]=temp;
      tempint=n_a[1];
      n_a[1]=n_a[0];
      n_a[0]=tempint;
    }
  }
  if(size>2){
    last=a[size-1]; 
    for(i=0;i<=k;i++){
      if(a[i]<=last){
	      minsize++;
        }
      else{
	      while(k>i && a[i]>last){
	        if(a[k]<=last){
	          minsize++;
	          temp=a[i];
	          a[i]=a[k];
	          a[k]=temp;
	          tempint=n_a[i];
	          n_a[i]=n_a[k];
	          n_a[k]=tempint;
	        }
	        else k--;
	      }
      }
    }
    if(a[size-1]==last){
      quicksort(a,n_a,size-1);
    }
    else{
      quicksort(a,n_a,minsize);
      quicksort(a+minsize,n_a+minsize,size-minsize);
    }
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

struct vector rotate(struct vector input, int modulo) {
  struct vector output;
  int i;

  for (i = 0; i < modulo; i++)
    output.n[i] = (input.n[i] + 1) % modulo;
  return output;
}

//
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
    var1 = (int)floor(FRANDOM * N);                          //connect node var1 with var2
    do {
      var2 = (int)floor(FRANDOM * N);                      
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
    } while (var1 == var2); 
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


//Compute the overlap with the planted solution
//Problem is: there is a symmetry between permutation of colors, so every permutation has to be checked
double overlapPlanted(int *color){
  int i, j, overlap, maxOver=0;                 //overlap computed, and max over all overlaps
  for (i = 0; i < Q; i++)
    group[i] = zero;                            //init the array of struct group with the struct zero in each position
  for (i = 0; i < N; i++)
    group[(int)(i/NoverQ)].n[color[i]]++;      
  for (i = 0; i < fact[Q]; i++) {
    overlap = 0;
    for (j = 0; j < Q; j++)
      overlap += group[j].n[perm[i].n[j]];
    if (overlap > maxOver) maxOver = overlap;
  }
  return (double)(Q*maxOver-N)/(Q-1)/N;
}

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

//Assign the color with the max prob in theta, being theta shape = [Q]
int assign_color(double *theta){
  double max=-1;
  int i,n_max=1,i_max[Q];               //i_max is the color with the max probability
  for(i=0;i<Q;i++){
    if(theta[i]>max){
      n_max=1;
      i_max[0]=i;
      max=theta[i];
    }else{
      if(theta[i]==max){
	      i_max[n_max]=i;                 //if two or more colors have the same prob, save all, later will pick a random color among them
	      n_max++;
      }
    }
  }
  if(n_max==1){
    return i_max[0];
  }else{
    return i_max[(int) floor(FRANDOM*n_max)]; //if there are two or more colors pick a random one
  }
}

//assign color probabilistically
int assign_color_prob(double *theta){
  double ran, cumulative;
  int q=0, color;
  cumulative = 0;
  ran = FRANDOM;
  while(cumulative<ran){
    color = q;
    cumulative += theta[q];
    q++;
  }
  return color;
}

//init random cavity marginals and put zero if prohibited color
void init_theta(double **theta_i, double ***theta, int **prohib_cols, int **decimated_nodes){     
  int i,j,q;
  double sum,ran;

  for(i=0;i<N;i++){
    sum=0;                                                //decimated_nodes[i][Q+1]==1 iif node decimated
    if(decimated_nodes[i][Q]==1){                       //if node i already decimated, probs already decided
      for(q=0; q<Q; q++){
        theta_i[i][q] = (float)decimated_nodes[i][q];
        sum += (float)decimated_nodes[i][q];
      }
    }else{
      for(q=0;q<Q;q++){
        /*if(prohib_cols[i][q]!=1){                         //check if col is prohibited
          ran=FRANDOM;
        }else{
          ran=0;
        }*/
        ran = FRANDOM;
        theta_i[i][q]=ran;                                //init random probabilities
        sum+=ran;
      }
    }
    //sure almost one is !=0 because checked in main
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

//Not used
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

//check if any variable has all colors prohibited
int check_contradiction(int **prohib_cols, int N, int *color){
  int contradiction, zero_probs, i;
  contradiction = 0;
  for(i=0; i<N; i++){
    zero_probs = 0;
    for(int q=0; q<Q; q++){
      if(prohib_cols[i][q]==0){
        zero_probs = 0;
        break;
      }else{
        zero_probs ++;
      }
    }
    if(zero_probs == Q){                    //if alla cols prohibited, contradiction=1
      contradiction += 1;
    }
  }
  return contradiction;
}

//populate the array abs_diff with the differences between max and second max probabilities for each node
void compute_abs_diff(double *abs_diff, double **theta_i, int *remaining_nodes, int remaining_variables){
  int i, maxQ, secondQ;
  for(int r=0; r<remaining_variables; r++){
    i = remaining_nodes[r];
    //decide if it's bigger first or second color
    if(theta_i[i][0]>=theta_i[i][1]){
      maxQ = 0;
      secondQ = 1;
    }else{
      maxQ = 1;
      secondQ = 0;
    }
    //check other colors from 2 to Q
    for(int q=2; q<Q; q++){
      if(theta_i[i][q]>theta_i[i][maxQ]){
        secondQ = maxQ;
        maxQ = q;
      }else if(theta_i[i][q]>theta_i[i][secondQ]){
        secondQ = q;
      }
    }
    //now compute the diff between first and second probs
    abs_diff[i] = theta_i[i][maxQ]-theta_i[i][secondQ];
  }
}

//do the decimation: write the decimated_nodes [N,Q+1] (1 for the color decided, and 1 at position Q+1) 
//and the prohib_cols [N,Q] (1 if neighbor is decimated in that color)
//write the colors in the color array
void decimate(int *to_decimate_nodes, int **decimated_nodes, int **prohib_cols, double **theta_i, int *color, int nDecim_temp){
  int i, *n_ch, *pos, connected_node;
  for(int z=0; z<nDecim_temp; z++){
    i = to_decimate_nodes[z];
    // ASSIGN MAX COLOR!!!
    color[i] = assign_color(theta_i[i]);     
    // ASSIGN COLOR PROBABILISTICALLY!!!
    //color[i] = assign_color_prob(theta_i[i]);        
    decimated_nodes[i][Q] = 1;                         //1 at decimated label (position Q+1)
    decimated_nodes[i][color[i]] = 1;                    //1 at color decided
    for(int j=0; j<deg[i]; j++){
      connected_node = neigh[i][j];
      prohib_cols[connected_node][color[i]] = 1;         //write 1 at prohibited color
    }
  }
}

//If all probs are equal to zero, entropic zero T limit
void find_min(int Nzero[Q], double *theta_new, int chosen, int j, double ***theta, int **prohib_cols){
  int i,n_min,i_min[Q],k, min=deg[chosen]+1;
  double z;
  for(int q=0; q<Q; q++){
    i_min[q] = q;
  }
  n_min = Q; //in this way, if all prohibited, extract a random color
  for(int q=0;q<Q;q++){                        //find the color with the min number of zeros
    if(prohib_cols[chosen][q]==0){
      if(Nzero[q]<min){
        n_min=1;
        i_min[0]=q;
        min=Nzero[q];
      }else{                                 //if more than one color has that min number of zeros, n_min>1 and i_min has more colors
        if(Nzero[q]==min){
	        i_min[n_min]=q;
	        n_min++;
        }
      }
    }
  }
  for(int q=0;q<Q;q++){                         //init at zero all probs
    theta_new[q]=0.;
  }
  if(n_min==1){                             //If only a color has the min number of zeros, i gets that color
    theta_new[i_min[0]]=1.;
  }else{                                    //If 2 or more colors have the same number of zeros:
    z=0.;
    for(int m=0;m<n_min;m++){                   //For each of that colors
      theta_new[i_min[m]]=1.;
      for(k=0;k<deg[chosen];k++){           //Normal cavity eq. but only for non zero contributions terms of that selected colors
	      if(k!=j){
	        if((1.-theta[neigh[chosen][k]][neigh_pos[chosen][k]][i_min[m]])!=0.0){
	          theta_new[i_min[m]]*=(1.-theta[neigh[chosen][k]][neigh_pos[chosen][k]][i_min[m]]); 
	        }
	      }
      }
      z+=theta_new[i_min[m]];
    }
    /*for(int m=0;m<n_min;m++){
      theta_new[i_min[m]]/=z;
    } */   
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
int iteration(int t_MAX,double **theta_i, double ***theta, int **prohib_cols, int *remaining_nodes, int remaining_variables){
  int t,r,i,j,k, node_to_choose, *n_ch,*pos,q,Nzero[Q],chosen_q;
  double diff=1,theta_new[Q],z,eps=1e-30;
  for(t=0;t<t_MAX&&diff>1e-7;t++){
    diff=0.; //f=0;
    for(r=0;r<remaining_variables;r++){                            // num of updates = remaining variables
      node_to_choose=(int)(FRANDOM*(double)remaining_variables);   //random node to update, in the remaining variables
      i = remaining_nodes[node_to_choose];                          //get the i value of the random chosen node wrt the remaining
      n_ch=neigh[i]; pos=neigh_pos[i];
      for(j=0;j<deg[i];j++){                                //update of theta[i][j][:]
	      z=0;
	      for(q=0;q<Q;q++){                                   //compute theta_{i->j}(s_i) and store it in theta_new[q]
          Nzero[q]=0;
          if(prohib_cols[i][q]==1){
            theta_new[q] = 0;
            Nzero[q] ++;                            //in this way find_min cannot chose q as the color to assign, except when all cols are prohibited
          }else{                                            
	          theta_new[q]=1;                                   //that is the marginal of i excluding the node j
	          for(k=0;k<deg[i];k++){                            //running over all neighbors k of i
	            if(k!=j){                                       //cavity marginal expression, at T=0:
	              theta_new[q]*=(1.-theta[n_ch[k]][pos[k]][q]); //entrambi divisi per d per avere un numero di ordine 1
	              if((1.-theta[n_ch[k]][pos[k]][q])==0.0){
		              Nzero[q]++;                                 //count the number of zeros chosing a certain color
                                                            //to make the entropic T=0 limit, useful iif 
	              }
	            }
	          }
	          z+=theta_new[q];                                  //z is the normalization; 
          }
 	      }
	      if(z!=0.0){                                         //if different from zero, it is possible to normalize
	        for(q=0;q<Q;q++){
	          theta_new[q]/=z;                                                   //convex combination to damp the algorithm
	          theta_new[q]=DAMP*theta[i][j][q]+(1.-DAMP)*theta_new[q];           //DAMP is defined as 0.5
	          diff+=fabs((theta_new[q]-theta[i][j][q])/(theta_new[q]+eps));      //diff between old and new
	          theta[i][j][q]=theta_new[q];   
	        }
	      }else{ //if all probs are equal to zero, entropic zero T limit with function find_min
	        find_min(Nzero,theta_new,i,j,theta, prohib_cols);                                 
	        for(q=0;q<Q;q++){
	          theta_new[q]=DAMP*theta[i][j][q]+(1.-DAMP)*theta_new[q];
	          diff+=fabs((theta_new[q]-theta[i][j][q])/(theta_new[q]+eps)); 
	          theta[i][j][q]=theta_new[q];
	        }
	      }         
      }
    }
  }

  for(i=0;i<N;i++){
    n_ch=neigh[i]; pos=neigh_pos[i];
    z=0;
    for(q=0;q<Q;q++){ //aggiorno theta[i]
      theta_new[q]=1;
      Nzero[q]=0;
      for(k=0;k<deg[i];k++){
	      theta_new[q]*=(1.-theta[n_ch[k]][pos[k]][q]); 
	      if((1.-theta[n_ch[k]][pos[k]][q])==0.0){
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
      find_min(Nzero,theta_new,i,-1,theta, prohib_cols);
      for(q=0;q<Q;q++){
	      theta_new[q]=DAMP*theta_i[i][q]+(1.-DAMP)*theta_new[q];
	      theta_i[i][q]=theta_new[q];   
      }
    }
  }

  return t;
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
  int i, j, nIter, is, nSamples, nDecim, nDecim_temp, t, remaining_variables, contradiction, itemp, c_num;
  int *color, *remaining_nodes, **prohib_cols, **decimated_nodes, *to_decimate_nodes;
  double c, c_min, c_max, delta_c, ***theta,**theta_i, *abs_diff;
  FILE *f;
  char file_name[100];

  //##############################################   Input control
  if (argc != 9 && argc!=8) {
    fprintf(stderr, "usage: %s <N> <c_min> <c_max> <delta_c> <nIter> <nSamples> [seed]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  N = atoi(argv[1]);
  c_min = atof(argv[2]);
  c_max = atof(argv[3]);
  delta_c = atof(argv[4]);
  nIter = atoi(argv[5]);
  nSamples = atoi(argv[6]);
  nDecim = 1;
  char *baseName = argv[7];

  if (argc == 9) {                                     //random selcted seed
    myrand = (unsigned)atoi(argv[8]);
    if (myrand == 2147483647)
      error("seed must be less than 2147483647");
  } else {                                             //otherwise random seed
    FILE *devran = fopen("/dev/random","r");         
    fread(&myrand, 4, 1, devran);
    fclose(devran);
  }

  //if (Q * (int)(N/Q) != N) 
  //  error("Q must divide N");
  if(N<nDecim){
    error("nDecim must be smaller than N");
  }     
  //##############################################    End input control

  theta=(double ***)malloc(N*sizeof(double**));
  theta_i=(double **)malloc(N*sizeof(double*));         
  color=(int *)malloc(N*sizeof(int));                   //color has a shape [N]
  prohib_cols = (int **)malloc(N*sizeof(int*));         //prohib_cols [N,Q], has zeros for available cols and 1 for not available
  decimated_nodes = (int **)malloc(N*sizeof(int*));     // [N,Q+1], has 1 in the color chosen for that node, and 1 in last position if decimated
  abs_diff=(double *)malloc(N*sizeof(double)); //shape [remaining_variables], for each node the abs difference between most probable variable and second 
  to_decimate_nodes = (int *)malloc(N*sizeof(int));    //list of nodes to decimate
  remaining_nodes = (int *)realloc(remaining_nodes, N*sizeof(int));

  for(i=0;i<N;i++){
    color[i] = -1;
    prohib_cols[i] = (int *)malloc(Q*sizeof(int));
    decimated_nodes[i] = (int *)malloc((Q+1)*sizeof(int));
  }

  for(i=0;i<N;i++){
    theta_i[i]=(double *)malloc(Q*sizeof(double));
  }

  initRandom();
  allocateMem();
  initPerm(Q);

  //printf("# Q = %i   N = %i   M = %i   c = %f   nIter = %i  nDecim = %i  seed = %u\n",
  //Q, N, M, c_min, nIter, nDecim, myrand);
  printf("# N M id E nIter\n");

    //#####################################                 ALGORITHM
  for (is = 0; is < nSamples; is++){
    for(c=c_min; c<=c_max; c+=delta_c){
      //          fprintf(stderr, "c= %f \n", c);
      NoverQ = (int)(N/Q);
      M = (int)(0.5 * c * N + 0.5);                         //number of links M to have a mean connectivity c
      graph = (int*)realloc(graph, 2*M*sizeof(int));  
      file_name[0] = "\0";
      snprintf(file_name, sizeof(file_name), "%sErdosRenyi_N_%d_M_%d_id_%d.txt", baseName, N, M, is + 1);
      f = fopen(file_name,"r");
      create_A(f);      
      nDecim = 1;
        contradiction = 0;                                    
        for(int i=0; i<N; i++){
          for(int q=0; q<Q; q++){
            prohib_cols[i][q] = 0;
            decimated_nodes[i][q] = 0;
          }
          decimated_nodes[i][Q] = 0;
        }
        //theta has to be resized depending on new graph
        theta=(double ***)realloc(theta, N*sizeof(double**));         //theta has a shape [N,deg[i],Q]
        for(i=0;i<N;i++){                                     //define theta_i and theta
          theta[i]=(double **)malloc(deg[i]*sizeof(double*));
          for(j=0;j<deg[i];j++){
  	        theta[i][j]=(double *)malloc(Q*sizeof(double));
          }
        }
        //decimation iteration
        remaining_variables = N;
        while(remaining_variables>0){
          //############# decide number of variables to decimate and init arrays
          itemp = 0;
          for(i=0; i<N; i++){
            if(decimated_nodes[i][Q]==0){  //check if node i is decimated
              remaining_nodes[itemp] = i;
              itemp++;
            }                   
          }
          if(remaining_variables<=nDecim){                    //last iteration can have less then nDecim variables to assign
            nDecim_temp = remaining_variables;
          }else{
            nDecim_temp = nDecim;
          }
          init_theta(theta_i, theta, prohib_cols, decimated_nodes);
          //############# end number of decimations and init arrays
          t=iteration(nIter, theta_i, theta, prohib_cols, remaining_nodes, remaining_variables);   //t of convergence or nIter if reached
          //After computing all probabilities theta:   
          compute_abs_diff(abs_diff, theta_i, remaining_nodes, remaining_variables);   //for every node, its |maxProb-secondProb|        
          quicksort(abs_diff, remaining_nodes, remaining_variables);                   //sort remaining_nodes depending on abs_diff
          for(int id=0; id<nDecim_temp; id++){
            to_decimate_nodes[id] = remaining_nodes[remaining_variables-id-1];         //pick last nDecim_temp nodes (that are the biggest)
          }       
          decimate(to_decimate_nodes, decimated_nodes, prohib_cols, theta_i, color, nDecim_temp);
          //CHECK IF THERE IS A CONTRADICTION: 0 no contradiction, 1 contradiction
          remaining_variables -= nDecim_temp;
        }
        contradiction = check_contradiction(prohib_cols, N, color);
      //End decimation process
      printf("%d %d %d %d %d\n", N, M, is+1, contradiction, nIter); 
      fflush(stdout);     
      freeMem(theta);
  } 
  //fprintf(stderr, "c= %f\n", c);
  }
  
  //#######################################
  return EXIT_SUCCESS;
}
