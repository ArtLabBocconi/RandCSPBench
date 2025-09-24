/* by FRT */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>


#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)

#define Q 3


struct node {
  int color, deg;
  struct node ** pNeigh;
  struct edge ** pEdge;
} *node;

struct edge {
  struct node * pNode[2];
  int whereIs;
} *edge, **unsatEdge;

int N, M, *graph, numUnsat;

/* variabili globali per il generatore random */
unsigned myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

unsigned randForInit(void) {
  unsigned long long y;
  
  y = myrand * 16807LL;
  myrand = (y & 0x7fffffff) + (y >> 31);
  if (myrand & 0x80000000) {
    myrand = (myrand & 0x7fffffff) + 1;
  }
  return myrand;
}

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


void error(char *string) {
  fprintf(stderr, "ERROR: %s\n", string);
  exit(EXIT_FAILURE);
}

void allocateMem(void) {
  int i;
  
  graph = (int *)calloc(2*M, sizeof(int));
  node = (struct node *)calloc(N, sizeof(struct node));
  edge = (struct edge *)calloc(M, sizeof(struct edge));
  unsatEdge = (struct edge **)calloc(M, sizeof(struct edge *));
}



void readGraph(char *filename) {
  int i, var1, var2;
  struct node * pn;
  struct node * pn1;
  FILE *f;
  f = fopen(filename, "r");

  fscanf(f, "N %d\n", &N);
  fscanf(f, "M %d\n", &M);

  allocateMem();

  for (pn = node; pn < node + N; pn++)
    pn->deg = 0;
  
  for (i = 0; i < M; i++) {
    fscanf(f,"e %d %d\n", &var1, &var2);
    graph[2*i] = var1 - 1;
    graph[2*i+1] = var2 - 1;
    node[var1 - 1].deg++;
    node[var2 - 1].deg++;
  }
  fclose(f);

  for (pn = node; pn < node + N; pn++) {
    pn->pNeigh = (struct node **)calloc(pn->deg, sizeof(struct node *));
    pn->pEdge = (struct edge **)calloc(pn->deg, sizeof(struct edge *));
    pn->deg = 0;
  }
  for (i = 0; i < M; i++) {
    pn = node + graph[2*i];
    pn1 = node + graph[2*i+1];
    edge[i].pNode[0] = pn;
    edge[i].pNode[1] = pn1;
    pn->pNeigh[pn->deg] = pn1;
    pn->pEdge[pn->deg] = edge + i;
    pn->deg++;
    pn1->pNeigh[pn1->deg] = pn;
    pn1->pEdge[pn1->deg] = edge + i;
    pn1->deg++;
  }
}


void initColors(void) {
  int i;
  
  for (i = 0; i < N; i++)
    node[i].color = (int)(FRANDOM * Q);
  //node[i].color = (int)(i / NoverQ);
}

void initUnsatEdges(void) {
  struct edge *pe;

  numUnsat = 0;
  for (pe = edge; pe < edge + M; pe++)
    if (pe->pNode[0]->color == pe->pNode[1]->color) {
      pe->whereIs = numUnsat;
      unsatEdge[numUnsat++] = pe;
    } else
      pe->whereIs = -1;
}

void pushIn(struct edge * pe) {
  pe->whereIs = numUnsat;
  unsatEdge[numUnsat++] = pe;
}

void takeOut(struct edge * pe) {
  assert(unsatEdge[pe->whereIs] == pe);
  unsatEdge[pe->whereIs] = unsatEdge[--numUnsat];
  unsatEdge[pe->whereIs]->whereIs = pe->whereIs;
  pe->whereIs = -1;
}

void freeMem(void) {
  struct node * pn;
  
  for (pn = node; pn < node + N; pn++) {
    free(pn->pNeigh);
    free(pn->pEdge);
  }
}



int main(int argc, char *argv[]) {
  clock_t begin = clock();
  int i, lowestUnsat, id;
  long long int t, maxIter, maxFlips, timeLowest;
  int oldColor, newColor, deltaUnsat;
  double c, eta, temperature;
  struct edge * pe;
  struct node * pn;
  // FILE *devran = fopen("/dev/urandom","r");
  // fread(&myrand, 4, 1, devran);
  // fclose(devran);
  myrand = 1;

  if (argc != 5) {
    fprintf(stderr, "graph from file: %s <eta> <maxIter> <filename> <id> \n", argv[0]);
    exit(EXIT_FAILURE);
  }

  eta = atof(argv[1]);
  maxIter = atoll(argv[2]);

  char filename[200];

  sscanf(argv[3], "%s", filename);
  id = atoi(argv[4]);

  readGraph(filename);
  c = 2.0 * M  / N;


  if (eta < 0.0 || eta >= 1.0) error("eta must be in [0,1)");
  if (eta == 0.0)
    temperature = 0.0;
  else
    temperature = -1./log(eta);
  maxFlips = maxIter * N;
  initRandom();
  
  initColors();
  initUnsatEdges();
  lowestUnsat = numUnsat;
  timeLowest = 0;
  t = 0;
  while (numUnsat && t < maxFlips) {
    pe = unsatEdge[(int)(FRANDOM * numUnsat)];
    assert(pe->pNode[0]->color == pe->pNode[1]->color);
    if (FRANDOM < 0.5)
	    pn = pe->pNode[0];
    else
	    pn = pe->pNode[1];
    oldColor = pn->color;
    newColor = (oldColor + 1 + (int)(FRANDOM * (Q-1))) % Q;
    deltaUnsat = 0;
    for (i = 0; i < pn->deg; i++)
	    deltaUnsat += (pn->pNeigh[i]->color == newColor) - (pn->pNeigh[i]->color == oldColor);
    if (deltaUnsat <= 0 || deltaUnsat < -temperature * log(FRANDOM)) {
	    for (i = 0; i < pn->deg; i++) {
	      if (pn->pNeigh[i]->color == oldColor)
	        takeOut(pn->pEdge[i]);
	      if (pn->pNeigh[i]->color == newColor)
	        pushIn(pn->pEdge[i]);
	    }
	    pn->color = newColor;
    }
    t++;
    if (numUnsat < lowestUnsat) {
	    lowestUnsat = numUnsat;
	    timeLowest = t;
    }
  }

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
  
  printf("%i %i %i %i %g %g %Lf %g\n", N, M, id, lowestUnsat, c, eta, (long double)timeLowest/N, time_spent);
  fflush(stdout);
  freeMem();

  
  return EXIT_SUCCESS;
}
