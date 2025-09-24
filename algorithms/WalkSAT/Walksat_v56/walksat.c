/* walksat */
/* Maintained by Henry Kautz <henry.kautz@gmail.com> */
#define VERSION "walksat version 56 May 2018"

/************************************/
/* Compilation flags                */
/************************************/

/********************************************************************/
/* Following tests set exactly one of the following flags to 1:     */
/*    BSD:   BSD Unix                                               */
/*    OSX:   Apple OS X                                           */
/*    LINUX: Linux Unix                                           */
/*    WINDOWS: Windows and DOS. Linking requires -l Winmm.lib       */
/*    POSIX: Other POSIX OS                                         */
/* Platform dependent differences:                                  */
/*    -arc4 only available on BSD and OSX. Option uses slower but   */
/*        stronger random number generation arc4random and          */
/*        arc4random_uniform                                        */
/*    WINDOWS and POSIX use rand() instead of random()              */
/*    Clock ticks per second determined by sysconf(_SC_CLK_TCK)     */
/*        for BSD, OSX, and LINUX                                   */
/*    Clock ticks per second fixed at 1000 for Windows              */
/*    Clock ticks per second fixed at 1 for POSIX                   */
/********************************************************************/

#if __FreeBSD__ || __NetBSD__ || __OpenBSD__ || __bsdi__ || _SYSTYPE_BSD
#define BSD 1
#elif __APPLE__ && __MACH__
#define OSX 1
#elif __unix__ || __unix || unix || __gnu_linux__ || linux || __linux
#define LINUX 1
#elif _WIN64 || _WIN23 || _WIN16 || __MSDOS__ || MSDOS || _MSDOS || __DOS__ 
#define NT 1
#else
#define POSIX 1
#endif

/* Define BIGINT to be the type used for the "cutoff" variable.
   Under gcc "long long int" gives a 64 bit integer.
   Under Windows __int64 gives a 64 bit integer.
   No way under POSIX to guarantee long int is 64 bits.
   Program will still function using a 32-bit value, but it
   limit size of cutoffs that can be specified. */

#if BSD || OSX || LINUX
#define BIGINT long long int
#define BIGFORMAT "lli"
#elif WINDOWS
#define BIGINT __int64
#define BIGFORMAT "I64d"
#elif POSIX
#define BIGINT long int
#define BIGFORMAT "li"
#endif

/************************************/
/* Standard includes                */
/************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <limits.h>
#include <signal.h>

#if BSD || OSX || LINUX
  #include <sys/times.h>
  #include <sys/time.h>
  #include <unistd.h>
#elif WINDOWS
  #include <time.h>
  #include <windows.h>
  #include <mmsystem.h>
#elif POSIX
  #include <sys/time.h>
#endif

#if POSIX || WINDOWS
  #define random() rand()
  #define srandom(seed) srand(seed)
#endif

/************************************/
/* Constant parameters              */
/************************************/

#define MAXFILENAME 2048
#define TRUE 1
#define FALSE 0
#define BIG 1000000000           /* a number bigger that the possible number of violated clauses */

#define HISTMAX 64		/* length of histogram of tail */
#define MAXATTEMPT 10           /* max number of times to attempt to find a non-tabu variable to flip */
#define denominator 100000       /* denominator used in fractions to represent probabilities */
#define ONE_PERCENT 1000         /* ONE_PERCENT / denominator = 0.01 */

/* #define DEBUG */

/************************************/
/* Main data structures             */
/************************************/

/* Atoms start at 1 */
/* Not a is recorded as -1 * a */
/* One dimensional arrays are statically allocated. */
/* Two dimensional arrays are dynamically allocated in */
/* the second dimension only.  */

int numatom;                    /* number of atoms */
int numclause;                  /* number of clauses */
int numliterals;                /* number of instances of literals across all clauses */

int numfalse;			/* number of false clauses */
int numfreebie;                 /* number of freebies */


/* Data structures for clauses */

int ** clause;			/* clauses to be satisfied */
				/* indexed as clause[clause_num][literal_num] */
int * size;			/* length of each clause */
int * false;			/* clauses which are false */
int * lowfalse;                 /* clauses that are false in the best solution found so far */
int * wherefalse;		/* where each clause is listed in false */
int * numtruelit;		/* number of true literals in each clause */
int longestclause;

/* Data structures for atoms: arrays of size numatom+1 indexed by atom */

int *atom;		    /* value of each atom */
int *lowatom;               /* value of best state found so far */
int *solution;              /* value of solution */
BIGINT *changed;	    /* step at which atom was last flipped */
int *breakcount;	    /* number of clauses that become unsat if var if flipped */
int *makecount;	            /* number of clauses that become sat if var if flipped */
int *freebielist;           /* list of freebies */
int *wherefreebie;          /* where atom appears in freebies list, -1 if it does not appear */

/* Data structures literals: arrays of size 2*numatom+1, indexed by literal+numatom */

int **occurrence;      	        /* where each literal occurs, size 2*numatom+1            */
				/* indexed as occurrence[literal+numatom][occurrence_num] */

int *numoccurrence;     	/* number of times each literal occurs, size 2*numatom+1  */
				/* indexed as numoccurrence[literal+numatom]              */

/* Data structures for lists of clauses used in heuristics */

int * best;
int * besttabu;
int *any;

/************************************/
/* Global flags and parameters      */
/************************************/

/* Options */

FILE * cnfStream;
int status_flag = 0;		/* value returned from main procedure */
int abort_flag;
int heuristic;		/* heuristic to be used */

int numerator;	        /* make random flip with numerator/denominator frequency */
int arc4 = FALSE;
double walk_probability = 0.5;
int plus_flag = FALSE; /* for novelty heuristics */
int tabu_length;		/* length of tabu list */
BIGINT numflip;		/* number of changes so far */
int numrun = 10;
BIGINT cutoff = 100000;
BIGINT base_cutoff = 100000;
int target = 0;
int numtry = 0;			/* total attempts at solutions */
int numsol = BIG;	                /* stop after this many tries succeeds */
int superlinear = FALSE;
int makeflag = FALSE;		/* set to true by heuristics that require the make values to be calculated */
char initfile[MAXFILENAME] = { 0 };
int initoptions = FALSE;

int nofreebie = FALSE;
int maxfreebie = FALSE;
int freebienoise = 0;
double freebienoise_prob = 0.0;

int alternate_greedy = -1;
int alternate_walk = -1;
int alternate_greedy_state = FALSE;
int alternate_run_remaining = 0;

int adaptive = FALSE;         /* update noise level adaptively during search */
int stagnation_timer;         /* number of remaining flips until stagnation check is performed */
int last_adaptive_objective;  /* number of unsat clauses last time noise was adaptively updated */
double adaptive_phi;
double adaptive_theta;

/* Random seed */

unsigned int seed;  /* Sometimes defined as an unsigned long int */

#if BSD || LINUX || OSX
  struct timeval tv;
  struct timezone tzp;
  long ticks_per_second;
#elif NT
  DWORD win_time;     /* elapsed time in ms, since windows boot up */
#endif

/* Histogram of tail */

BIGINT tailhist[HISTMAX];	/* histogram of num unsat in tail of run */
long histtotal;
int tail = 10;
int tail_start_flip;
int undo_age = 1;
BIGINT undo_count;

/* Printing options */

int printonlysol = FALSE;
int printsolcnf = FALSE;
int printfalse = FALSE;
int printlow = FALSE;
int printhist = FALSE;
int printtrace = FALSE;
int trace_assign = FALSE;
char outfile[MAXFILENAME] = { 0 };

/* Statistics */

double expertime;
BIGINT flips_this_solution;
int lowbad;		        /* lowest number of bad clauses during try */
BIGINT totalflip = 0;		/* total number of flips in all tries so far */
BIGINT totalsuccessflip = 0;	/* total number of flips in all tries which succeeded so far */
int numsuccesstry = 0;		/* total found solutions */
BIGINT x;
BIGINT integer_sum_x = 0;
double sum_x = 0.0;
double sum_x_squared = 0.0;
double mean_x;
double second_moment_x;
double variance_x;
double std_dev_x;
double std_error_mean_x;
double seconds_per_flip;
int r;
int sum_r = 0;
double sum_r_squared = 0.0;
double mean_r;
double variance_r;
double std_dev_r;
double std_error_mean_r;
double avgfalse;
double sumfalse;
double sumfalse_squared;
double second_moment_avgfalse, variance_avgfalse, std_dev_avgfalse, ratio_avgfalse;
double std_dev_avgfalse;
double f;
double sample_size;
double sum_avgfalse = 0.0;
double sum_std_dev_avgfalse = 0.0;
double mean_avgfalse;
double mean_std_dev_avgfalse;
int number_sampled_runs = 0;
double ratio_mean_avgfalse;
double suc_sum_avgfalse = 0.0;
double suc_sum_std_dev_avgfalse = 0.0;
double suc_mean_avgfalse;
double suc_mean_std_dev_avgfalse;
int suc_number_sampled_runs = 0;
double suc_ratio_mean_avgfalse;
double nonsuc_sum_avgfalse = 0.0;
double nonsuc_sum_std_dev_avgfalse = 0.0;
double nonsuc_mean_avgfalse;
double nonsuc_mean_std_dev_avgfalse;
int nonsuc_number_sampled_runs = 0;
double nonsuc_ratio_mean_avgfalse;

/* Hamming calculations */

char hamming_target_file[MAXFILENAME] = { 0 };
char hamming_data_file[MAXFILENAME] = { 0 };
int hamming_sample_freq;
int hamming_flag = FALSE;
int hamming_distance;
int * hamming_target;
FILE * hamming_fp;

/**************************************/
/* Inline utility functions           */
/**************************************/

static inline int ABS(int x) { return x<0 ? -x : x ; }

static inline int RANDMOD(int x) {
#if OSX || BSD
    return x > 1 ? (arc4 ? arc4random_uniform((uint32_t) x) : random()%x) : 0;
#else
    return x > 1 ? random()%x : 0;
#endif
}


static inline int MAX(int x, int y) { return x > y ? x : y; }

static inline int MIN(int x, int y) { return x < y ? x : y; }

static inline int onfreebielist(int v) { return wherefreebie[v] != -1; }

static inline void addtofreebielist(int v)
{
    freebielist[numfreebie] = v;
    wherefreebie[v] = numfreebie++;
}

static inline void removefromfreebielist(int v)
{
    int swapv;
    int wherev;
	
    if (numfreebie < 1 || wherefreebie[v] < 0) {
	fprintf(stderr, "Freebie list error!\n");
	exit(-1);
    }
    numfreebie--;
    wherev = wherefreebie[v];
    wherefreebie[v] = -1;
    if (wherev == numfreebie) return;
    swapv = freebielist[numfreebie];
    freebielist[wherev] = swapv;
    wherefreebie[swapv] = wherev;
}

/************************************/
/* Forward declarations             */
/************************************/

void parse_parameters(int argc,char *argv[]);
void print_parameters(int argc, char * argv[]);

int pickrandom(void);
int pickbest(void);
int picktabu(void);
int picknovelty(void);
int pickrnovelty(void);
int pickalternate(void);
int pickbigflip(void);
int pickgsat(void);

enum heuristics { RANDOM, BEST, TABU, NOVELTY, RNOVELTY, 
		  ALTERNATE, BIGFLIP, GSAT};
static int (*pickcode[])(void) =
   {pickrandom, pickbest, picktabu,
    picknovelty, pickrnovelty,
    pickalternate, pickbigflip, pickgsat};

double elapsed_seconds(void);
int countunsat(void);
void scanone(int argc, char *argv[], int i, int *varptr);
void scanonell(int argc, char *argv[], int i, BIGINT *varptr);
void scanoned(int argc, char *argv[], int i, double *varptr);
void init(void);
void initprob(void); 
void flipatom(int toflip);
void print_false_clauses(int lowbad);
void save_false_clauses(int lowbad);
void print_low_assign(int lowbad);
void save_low_assign(void);
void save_solution(void);
void print_current_assign(void);
void handle_interrupt(int sig);
long super(int i);
void print_sol_file(char * filename);
void print_statistics_header(void);
void initialize_statistics(void);
void update_statistics_start_try(void);
void print_statistics_start_flip(void);
void update_and_print_statistics_end_try(void);
void update_statistics_end_flip(void);
void print_statistics_final(void);
void print_sol_cnf(void);
void read_hamming_file(void);
void open_hamming_data(void);
int calc_hamming_dist(int atom[], int hamming_target[], int numatom);



/************************************/
/* Main                             */
/************************************/



int main(int argc,char *argv[])
{
    int a;

#if BSD || LINUX || OSX
    ticks_per_second = sysconf(_SC_CLK_TCK);
    gettimeofday(&tv,&tzp);
    seed = (unsigned int)((( tv.tv_sec & 0177 ) * 1000000) + tv.tv_usec);
#elif WINDOWS
    seed = (unsigned int)(timeGetTime());
#elif POSIX
    seed = (unsigned int)(time());
#endif
    parse_parameters(argc, argv);
    srandom(seed);
    print_parameters(argc, argv);
    initprob();
    initialize_statistics();
    print_statistics_header();
    signal(SIGINT, handle_interrupt);
    abort_flag = FALSE;
    (void) elapsed_seconds();

    while (! abort_flag && numsuccesstry < numsol && numtry < numrun) {
	numtry++;
	init();
	update_statistics_start_try();
	numflip = 0;
	if (superlinear) cutoff = base_cutoff * super(numtry);

	while((numfalse > target) && (numflip < cutoff)) {
	    print_statistics_start_flip();
	    numflip++;

	    if (maxfreebie && numfreebie>0 && (freebienoise == 0 || RANDMOD(denominator) > freebienoise))
		a = freebielist[RANDMOD(numfreebie)];
	    else
		a = (pickcode[heuristic])();
	    flipatom(a);
	    update_statistics_end_flip();
	}
	update_and_print_statistics_end_try();
    }
    expertime = elapsed_seconds();
    print_statistics_final();
    return status_flag;
}


void flipatom(int toflip)
{
    int i, j;			
    int toenforce;		
    register int cli;
    register int lit;
    int numocc;
    register int sz;
    register int * litptr;
    int * occptr;
    int temp;

    if (numflip - changed[toflip] <= undo_age)
	undo_count++;
    
    changed[toflip] = numflip;
    if(atom[toflip] > 0)
      toenforce = -toflip;
    else
      toenforce = toflip;

    atom[toflip] = 1-atom[toflip];

    if (hamming_flag){
	if (atom[toflip] == hamming_target[toflip])
	  hamming_distance--;
	else
	  hamming_distance++;
	if ((numflip % hamming_sample_freq) == 0)
	  fprintf(hamming_fp, "%" BIGFORMAT " %i\n", numflip, hamming_distance);
    }

    numocc = numoccurrence[numatom-toenforce];
    occptr = occurrence[numatom-toenforce];
    for(i = 0; i < numocc ;i++)
      {
	  /* cli = occurrence[numatom-toenforce][i]; */
	  cli = *(occptr++);

	  if (--numtruelit[cli] == 0){
	      false[numfalse] = cli;
	      wherefalse[cli] = numfalse;
	      numfalse++;
	      /* Decrement toflip's breakcount */
	      breakcount[toflip]--;

	      if (maxfreebie){
		  if (breakcount[toflip] == 0 && makecount[toflip]>0 && ! onfreebielist(toflip))
		      addtofreebielist(toflip);
	      }

	      if (makeflag){
		/* Increment the makecount of all vars in the clause */
		sz = size[cli];
		litptr = clause[cli];
		for (j=0; j<sz; j++){
		  /* lit = clause[cli][j]; */
		  lit = *(litptr++);
		  makecount[ABS(lit)]++;
		  if (maxfreebie){
		      if (breakcount[ABS(lit)] == 0 && ! onfreebielist(ABS(lit)))
			  addtofreebielist(ABS(lit));
		  }
		}
	      }
	  }
	  else if (numtruelit[cli] == 1){
	      /* Find the lit in this clause that makes it true, and inc its breakcount */
	      litptr = clause[cli];
	      while (1){
		  /* lit = clause[cli][j]; */
		  lit = *(litptr++);
		  if((lit > 0) == atom[ABS(lit)]){
		      breakcount[ABS(lit)]++;

		      if (maxfreebie){
			  if (onfreebielist(ABS(lit)))
			      removefromfreebielist(ABS(lit));
		      }

		      /* Swap lit into first position in clause */
		      if ((--litptr) != clause[cli]) {
			      temp = clause[cli][0];
			      clause[cli][0] = *(litptr);
			      *(litptr) = temp;
			  }
		      break;
		  }
	      }
	  }
      }
    
    numocc = numoccurrence[numatom+toenforce];

    occptr = occurrence[numatom+toenforce];
    for(i = 0; i < numocc; i++)
      {
	  /* cli = occurrence[numatom+toenforce][i]; */
	  cli = *(occptr++);

	  if (++numtruelit[cli] == 1){
	      numfalse--;
	      false[wherefalse[cli]] =
		false[numfalse];
	      wherefalse[false[numfalse]] =
		wherefalse[cli];
	      /* Increment toflip's breakcount */
	      breakcount[toflip]++;

	      if (maxfreebie){
		  if (onfreebielist(toflip))
		      removefromfreebielist(toflip);
	      }

	      
	      if (makeflag){
		/* Decrement the makecount of all vars in the clause */
		sz = size[cli];
		litptr = clause[cli];
		for (j=0; j<sz; j++){
		  /* lit = clause[cli][j]; */
		  lit = *(litptr++);
		  makecount[ABS(lit)]--;

		  if (maxfreebie){
		      if (onfreebielist(ABS(lit)) && makecount[ABS(lit)]==0)
			  removefromfreebielist(ABS(lit));
		  }
		}
	      }
	  }
	  else if (numtruelit[cli] == 2){
	      /* Find the lit in this clause other than toflip that makes it true,
		 and decrement its breakcount */
	      litptr = clause[cli];
	      while (1){
		  /* lit = clause[cli][j]; */
		  lit = *(litptr++);
		  if( ((lit > 0) == atom[ABS(lit)]) &&
		     (toflip != ABS(lit)) ){
		      breakcount[ABS(lit)]--;

		      if (maxfreebie){
			  if (breakcount[ABS(lit)]==0 && makecount[ABS(lit)]>0 && !onfreebielist(ABS(lit)))
			      addtofreebielist(ABS(lit));
		      }
		      break;
		  }
	      }
	  }
      }
}




/************************************/
/* Initialization                   */
/************************************/


void parse_parameters(int argc,char *argv[])
{
    int i;
    int temp;

    heuristic = BEST;
    cnfStream = stdin;
    for (i=1;i < argc;i++)
    {
	/* General parameters */

	if (argv[i][0] != '-' && cnfStream == stdin){
            cnfStream = fopen(argv[i],"r");
            if (cnfStream == NULL){
                fprintf(stderr, "Cannot open file named %s\n", argv[i]);
                exit(-1);
            }
            ++i;
        }
        else if (strcmp(argv[i],"-seed") == 0){
	    scanone(argc,argv,++i,&temp);
            seed = (unsigned int)temp;
        }
	else if (strcmp(argv[i],"-status") == 0)
	    status_flag = 1;
	else if (strcmp(argv[i],"-cutoff") == 0)
	    scanonell(argc,argv,++i,&cutoff);
	else if (strcmp(argv[i],"-numsol") == 0)
	    scanone(argc,argv,++i,&numsol);
	else if (strcmp(argv[i],"-tries") == 0 || strcmp(argv[i],"-restart") == 0)
	    scanone(argc,argv,++i,&numrun);
	else if (strcmp(argv[i],"-target") == 0)
	    scanone(argc,argv,++i,&target);
	else if (strcmp(argv[i],"-arc4") == 0){
#if OSX || BSD
	    arc4 = TRUE;
#else
	    fprintf(stderr, "Sorry, cannot use -arc4 option on this system\n");
	    exit(-1);
#endif		
	}
	/* Printing */

	else if (strcmp(argv[i],"-out") == 0 && i<argc-1)
	    strcpy(outfile, argv[++i]);
	else if (strcmp(argv[i],"-hist") == 0)
	    printhist = TRUE;
	else if ((strcmp(argv[i],"-undo") == 0))
	    scanone(argc,argv,++i,&undo_age);
	else if (strcmp(argv[i],"-hamming") == 0  && i < argc-3){
	    sscanf(argv[++i], " %s", hamming_target_file);
	    sscanf(argv[++i], " %s", hamming_data_file);
	    sscanf(argv[++i], " %i", &hamming_sample_freq);
	    hamming_flag = TRUE;
	    numrun = 1;
	}
	else if (strcmp(argv[i],"-low") == 0)
	    printlow = TRUE;
	else if (strcmp(argv[i],"-sol") == 0) {
	    printonlysol = TRUE;
	    printlow = TRUE;
	}
	else if (strcmp(argv[i],"-solcnf") == 0) {
	    printsolcnf = TRUE;
	    numsol = 1;
	}
	else if (strcmp(argv[i],"-bad") == 0)
	    printfalse = TRUE;
	else if (strcmp(argv[i],"-trace") == 0)
	    scanone(argc,argv,++i,&printtrace);
	else if (strcmp(argv[i],"-assign") == 0){
	    scanone(argc,argv,++i,&printtrace);
	    trace_assign = TRUE;
	}
	else if (strcmp(argv[i],"-tail") == 0)
	    scanone(argc,argv,++i,&tail);
	
	/* Options for heuristics */

	else if (strcmp(argv[i],"-noise") == 0){
	    scanone(argc,argv,++i,&numerator);
	    if (i < argc-1 && sscanf(argv[i+1],"%i",&temp)==1){
		walk_probability = ((double)numerator)/temp;
		i++;
	    }
	    else {
		walk_probability = ((double)numerator)/100.0;
	    }
	}
	else if (strcmp(argv[i],"-walkprob") == 0 || strcmp(argv[i],"-wp") == 0)
	    scanoned(argc,argv, ++i, &walk_probability);
	else if (strcmp(argv[i],"-init") == 0  && i < argc-1)
	    sscanf(argv[++i], " %s", initfile);
	else if (strcmp(argv[i],"-super") == 0){
	    superlinear = TRUE;
	}
	else if (strcmp(argv[i],"-nofreebie") == 0)
	    nofreebie = TRUE;
	else if (strcmp(argv[i],"-maxfreebie") == 0){
	    makeflag = TRUE;
	    maxfreebie = TRUE;
	    if (i < argc-1 && sscanf(argv[i+1],"%lg",&freebienoise_prob)==1){
		i++;
	    }
	    freebienoise = (int)(freebienoise_prob * denominator);
	}
        else if (strcmp(argv[i],"-adaptivehh") == 0){
	    /* start adaptive search at 0 noise */
	    walk_probability = 0.0;
	    adaptive = TRUE;
	    if (i < argc-1 && sscanf(argv[i+1],"%lg",&adaptive_phi)==1){
		i++;
	    }
	    else adaptive_phi = 0.20;
	    if (i < argc-1 && sscanf(argv[i+1],"%lg",&adaptive_theta)==1){
		i++;
	    }
	    else adaptive_theta = 0.20;
        }

	/* Heuristics */
	else if (strcmp(argv[i],"-random") == 0)
	    heuristic = RANDOM;
        else if (strcmp(argv[i],"-rnovelty") == 0){
            heuristic = RNOVELTY;
            makeflag = TRUE;
        }
        else if (strcmp(argv[i],"-novelty") == 0){
            heuristic = NOVELTY;
            makeflag = TRUE;
        }
	else if ((strcmp(argv[i],"-plus") == 0))
	    plus_flag = TRUE;
	else if (strcmp(argv[i],"-best") == 0 || strcmp(argv[i],"-walksat") == 0)
	    heuristic = BEST;
	else if (strcmp(argv[i],"-gsat") == 0){
	    heuristic = GSAT;
	    makeflag = TRUE;
	}
	else if (strcmp(argv[i],"-alternate") == 0){
	    scanone(argc,argv,++i,&alternate_walk);
	    scanone(argc,argv,++i,&alternate_greedy);
	    walk_probability = 0.0;
	    heuristic = ALTERNATE;
	}
	else if (strcmp(argv[i],"-bigflip") == 0){
	    scanone(argc,argv,++i,&alternate_walk);
	    scanone(argc,argv,++i,&alternate_greedy);
	    heuristic = BIGFLIP;
	}
	else if (strcmp(argv[i],"-tabu") == 0) {
	    scanone(argc,argv,++i,&tabu_length);
	    heuristic = TABU;
	    walk_probability = 0.0;
	}
	else {
	    fprintf(stderr, "General parameters:\n");
	    fprintf(stderr, "  filename = CNF file, if not specified read stdin\n");
	    fprintf(stderr, "  -seed N = use N to initialize random()\n");
#if BSD || OSX	    
	    fprintf(stderr, "  -arc4 = use arc4random_uniform random number generator; seed is ignored\n");
#endif
	    fprintf(stderr, "  -cutoff N = bound on the number of flips per trial\n");
	    fprintf(stderr, "      N can have suffix K (thousands), M (millions), or B (billions)\n");
	    fprintf(stderr, "  -restart N = bound on the number of trials\n");
	    fprintf(stderr, "  -numsol N = stop after finding N solutions\n");
	    fprintf(stderr, "  -status = return fail status if solution not found\n");
	    fprintf(stderr, "  -target N = succeed if N or fewer clauses unsatisfied\n");
	    fprintf(stderr, "Heuristics:\n");
	    fprintf(stderr, "  -random\n");
	    fprintf(stderr, "  -best = minimize breaks (default)\n");
	    fprintf(stderr, "  -gsat = minimize breaks - makes\n");
	    fprintf(stderr, "  -tabu N = tabu list length N; default walkprob = 0.0\n");
	    fprintf(stderr, "  -novelty = optional argument -plus\n");
	    fprintf(stderr, "  -rnovelty = optional argument -plus\n");
            fprintf(stderr, "  -alternate M N = alternate M walk steps with N greedy steps\n");
	    fprintf(stderr, "  -bigflip M N = after deciding between walk and greedy, make M walk or N greedy flips\n");
	    fprintf(stderr, "Options for heuristics:\n");
	    fprintf(stderr, "  -walkprob R or -wp R = walk probability (default R = 0.5)\n");
	    fprintf(stderr, "  -noise N or -noise N M = sets walk probability to N/M (default M = 100)\n");
            fprintf(stderr, "  -nofreebie = disable freebie rule\n");
            fprintf(stderr, "  -maxfreebie or -maxfreebie P = check for freebies before picking an unsat clause, skip with probability P (default P = 0.0)\n");
	    fprintf(stderr, "  -super N = use the series 1,1,2,1,1,2,4,... times cutoff for bound on flips per trial\n");
            fprintf(stderr, "  -adaptivehh or -adaptivehh PHI THETA = adjust noise level adaptively using Holger Hoos method (default PHI = 0.20 THETA = 0.20)\n");
	    fprintf(stderr, "  -init FILE = initialize literals specified in file, others set randomly\n");
	    fprintf(stderr, "Printing:\n");
	    fprintf(stderr, "  -out FILE = print solution as a list of literals to FILE\n");
	    fprintf(stderr, "  -trace N = print statistics every N flips\n");
	    fprintf(stderr, "  -assign N = print assignments at flip N, 2N, ...\n");
	    fprintf(stderr, "  -sol = print satisfying assignments to stdout\n");
	    fprintf(stderr, "  -solcnf = print sat assign to stdout in DIMACS format, and exit\n");
	    fprintf(stderr, "  -low = print lowest assignment each try\n");
	    fprintf(stderr, "  -bad = print unsat clauses each try\n");
	    fprintf(stderr, "  -hist = print histogram of tail\n");
	    fprintf(stderr, "  -tail N = assume tail begins at nvars*N (default N = 10)\n");
	    fprintf(stderr, "  -undo N = count a flip as undo if change is within N flips (default N = 1)\n");
	    fprintf(stderr, "  -hamming TARGET_FILE DATA_FILE SAMPLE_FREQUENCY\n");
	    exit(-1);
	}
    }
    base_cutoff = cutoff;
    if (numsol>numrun) numsol = numrun;
    numerator = (int)(walk_probability * denominator);
}



void init(void)
{
    int i;
    int j;
    int var;
    int thetruelit;
    FILE * infile;
    int lit;

    alternate_run_remaining = 0;
    alternate_greedy_state = FALSE;
    if (adaptive) {
	walk_probability = 0.0;
	numerator = (int)(walk_probability * denominator);
	stagnation_timer = (int)(numclause * adaptive_theta);
	last_adaptive_objective = BIG;
    }
    
    /* initialize truth assignment and changed time */
    for(i = 0;i < numclause;i++)
      numtruelit[i] = 0;
    numfalse = 0;
    for(i = 1;i < numatom+1;i++)
      {
	  changed[i] = -i - 1000;  /* ties in age between unchanged variables broken for lowest-numbered */
	  breakcount[i] = 0;
	  makecount[i] = 0;
	  atom[i] = RANDMOD(2);
      }

    if (initfile[0]){
	if ((infile = fopen(initfile, "r")) == NULL){
	    fprintf(stderr, "Cannot open %s\n", initfile);
	    exit(1);
	}
	i=0;
	while (fscanf(infile, " %i", &lit)==1){
	    i++;
	    if (ABS(lit)>numatom){
		fprintf(stderr, "Bad init file %s\n", initfile);
		exit(1);
	    }
	    if (lit<0) atom[-lit]=0;
	    else atom[lit]=1;
	}
	if (i==0){
	    fprintf(stderr, "Bad init file %s\n", initfile);
	    exit(1);
	}
	fclose(infile);
    }

    /* Initialize breakcount and makecount */
    for(i = 0;i < numclause;i++)
      {
	  for(j = 0;j < size[i];j++)
	    {
		if((clause[i][j] > 0) == atom[ABS(clause[i][j])])
		  {
		      numtruelit[i]++;
		      thetruelit = clause[i][j];
		  }
	    }
	  if(numtruelit[i] == 0)
	    {
		wherefalse[i] = numfalse;
		false[numfalse] = i;
		numfalse++;
		for(j = 0;j < size[i];j++){
		  makecount[ABS(clause[i][j])]++;
		}
	    }
	  else if (numtruelit[i] == 1)
	    {
		breakcount[ABS(thetruelit)]++;
	    }
      }

    /* Create freebie list */
    numfreebie = 0;
    for (var = 1; var<=numatom; var++)
	wherefreebie[var] = -1;
    for (var = 1; var<=numatom; var++){
	if (makecount[var] > 0 && breakcount[var] == 0){
	    wherefreebie[var] = numfreebie;
	    freebielist[numfreebie++] = var;
	}
    }

#ifdef DEBUG
    for (i = 0; i<numfreebie; i++)
	printf(" %d at %d \n", freebielist[i], wherefreebie[freebielist[i]]);
#endif

    /* Initialize hamming distance calculation */
    if (hamming_flag){
	hamming_distance = calc_hamming_dist(atom, hamming_target, numatom);
	fprintf(hamming_fp, "0 %i\n", hamming_distance);
    }
}


void initprob(void)
{
    int i;
    int j;
    int lastc;
    int nextc;
    int lit;
    int *storebase;
    int storesize;
    int storeused;
    int *storeptr;

    while ((lastc = getc(cnfStream)) == 'c') {
	while ((nextc = getc(cnfStream)) != EOF && nextc != '\n');
    }
    ungetc(lastc,cnfStream);
    if (fscanf(cnfStream,"p cnf %i %i",&numatom,&numclause) != 2){
	fprintf(stderr,"Bad input file\n");
	exit(-1);
    }

    clause = (int **) calloc(sizeof(int *), (numclause+1));
    size = (int *) calloc(sizeof(int), (numclause+1));
    false = (int *) calloc(sizeof(int), (numclause+1));
    lowfalse = (int *) calloc(sizeof(int), (numclause+1));
    wherefalse = (int *) calloc(sizeof(int), (numclause+1));
    numtruelit = (int *) calloc(sizeof(int), (numclause+1));

    occurrence = (int **) calloc(sizeof(int *), (2*numatom+1));
    numoccurrence = (int *) calloc(sizeof(int), (2*numatom+1));
    atom = (int *) calloc(sizeof(int), (numatom+1));
    lowatom = (int *) calloc(sizeof(int), (numatom+1));
    solution = (int *) calloc(sizeof(int), (numatom+1));
    changed = (BIGINT *) calloc(sizeof(BIGINT), (numatom+1));
    breakcount = (int *) calloc(sizeof(int), (numatom+1));
    makecount = (int *) calloc(sizeof(int), (numatom+1));
    freebielist = (int *) calloc(sizeof(int), (numatom+1));
    wherefreebie = (int *) calloc(sizeof(int), (numatom+1));
    hamming_target = (int *) calloc(sizeof(int), (numatom+1));
    
    numliterals = 0;
    longestclause = 0;

    /* Read in the clauses and set number of occurrences of each literal */
    storesize = 1024;
    storeused = 0;
    printf("Reading formula\n");
    storebase = (int *) calloc(sizeof(int), 1024);

    for(i = 0;i < 2*numatom+1;i++)
	numoccurrence[i] = 0;
    for(i = 0;i < numclause;i++) {
	size[i] = 0;
	do {
	    if (fscanf(cnfStream,"%i ",&lit) != 1) {
		fprintf(stderr, "Bad input file\n");
		exit(-1);
	    }
	    if(lit != 0) {
		if (storeused >= storesize){
		    storeptr = storebase;
		    storebase = (int *) calloc(sizeof(int), storesize*2);
		    for (j=0; j<storesize; j++)
			storebase[j] = storeptr[j];
		    free((void *) storeptr);
		    storesize *= 2;
		}
		size[i]++;
		storebase[storeused++] = lit;
		numliterals++;
		numoccurrence[lit+numatom]++;
	    }
	} while(lit != 0);
	if(size[i] == 0) {
	    fprintf(stderr,"Bad input file\n");
	    exit(-1);
	}
	longestclause = MAX(longestclause, size[i]);
    }

    printf("Creating data structures\n");
    
    /* Have to wait to set the clause[i] ptrs to the end, since store might move */
    j=0;
    for (i=0; i<numclause; i++){
	clause[i] = &(storebase[j]);
	j += size[i];
    }

    best = calloc(sizeof(int), longestclause);
    besttabu = calloc(sizeof(int), longestclause);
    any = calloc(sizeof(int), longestclause);

    /* Create the occurence lists for each literal */

    /* First, allocate enough storage for occurrence lists */
    storebase = (int *)calloc(sizeof(int), numliterals);

    /* printf("numliterals = %d\n", numliterals); fflush(stdout); */
    
    /* Second, allocate occurence lists */
    i = 0;
    for (lit = -numatom; lit <= numatom; lit++){

	/* printf("lit = %d    i = %d\n", lit, i); fflush(stdout); */
	
	if (lit != 0){
	    if (i>numliterals){
		fprintf(stderr, "Code error, allocating occurrence lists\n");
		exit(-1);
	    }
	    occurrence[lit+numatom] = &(storebase[i]);
	    i += numoccurrence[lit+numatom];
	    numoccurrence[lit+numatom] = 0;
	}
    }

    /* Third, fill in the occurence lists */
    for(i = 0; i < numclause; i++) {
	  for(j = 0; j < size[i]; j++) {
	      lit = clause[i][j];
	      occurrence[lit+numatom][numoccurrence[lit+numatom]] = i;
	      numoccurrence[lit+numatom]++;
	  }
    }
}




/************************************/
/* Printing and Statistics          */
/************************************/


void print_parameters(int argc, char * argv[])
{
  int i;

  printf("%s compiled for %s\n", VERSION,
#if OSX
	 "OSX"
#elif BSD
	 "BSD"
#elif LINUX
	 "Linux"
#elif WINDOWS
	 "Windows"
#else
	 "POSIX"
#endif	 
      );
  printf("command line =");
  for (i=0;i < argc;i++){
    printf(" %s", argv[i]);
  }
  printf("\n");
  printf("seed = %u\n",seed);
  printf("cutoff = %" BIGFORMAT "\n",cutoff);
  printf("tries = %i\n",numrun);
  printf("walk probabability = %5.3f\n", walk_probability);
  printf("\n");
}



void initialize_statistics(void)
{
    x = 0; r = 0;
    if (hamming_flag) {
	read_hamming_file();
	open_hamming_data();
    }
    tail_start_flip = tail * numatom;
    printf("tail starts after flip = %i\n", tail_start_flip);
}



void print_statistics_header(void)
{
    printf("numatom = %i, numclause = %i, numliterals = %i\n",numatom,numclause,numliterals);
    printf("wff read in\n\n");

    printf("    lowbad     unsat       avg   std dev    sd/avg     flips      undo              length       flips       flips\n");
    printf("      this       end     unsat       avg     ratio      this      flip   success   success       until         std\n");
    printf("       try       try      tail     unsat      tail       try  fraction      rate     tries      assign         dev\n\n");

    fflush(stdout);
}



void update_statistics_start_try(void)
{
    int i;

    lowbad = numfalse;
    undo_count = 0;

    sample_size = 0;
    sumfalse = 0.0;
    sumfalse_squared = 0.0;

    for (i=0; i<HISTMAX; i++)
	tailhist[i] = 0;
    if (tail_start_flip == 0){
	tailhist[numfalse < HISTMAX ? numfalse : HISTMAX - 1] ++;
    }
	      
    if (printfalse) save_false_clauses(lowbad);
    if (printlow) save_low_assign();
}



void print_statistics_start_flip(void)
{
    if (printtrace && (numflip % printtrace == 0)){
	printf(" %9i %9i                     %9" BIGFORMAT "\n", lowbad,numfalse,numflip);
	if (trace_assign)
	    print_current_assign();
	fflush(stdout);
    }
}


void update_statistics_end_flip(void)
{

    if (adaptive){
	/* Reference for adaptie noise option:
	   An Adaptive Noise Mechanism for WalkSAT (Corrected). Holger H. Hoos.
	*/
	
	if (numfalse < last_adaptive_objective) {
	    last_adaptive_objective = numfalse;
	    stagnation_timer = (int)(numclause * adaptive_theta);
	    /* p = p - p * (phi)/2
               p = (1 - phi/2) * p
               p = (1 - phi/2) * (numerator / denominator)
               p (denominator) = (1 - phi/2) * numerator
               numerator = (1 - phi/2) * numerator
	    */
	    numerator = (int)((1.0 - adaptive_phi / 2.0) * numerator);
	}
	else {
	    stagnation_timer = stagnation_timer - 1;
	    if (stagnation_timer <= 0){
		last_adaptive_objective = numfalse;
		stagnation_timer = (int)(numclause * adaptive_theta);
		/* p = p + (1 - p) * phi
		   denominator * p = denominator * p + denominator * (1 - p) * phi
 		   numerator = numerator + denominator * (1 - p) * phi;
 		   numerator = numerator + denominator * (1 - numerator/denominator) * phi;
 		   numerator = numerator + (denominator - numerator) * phi;
		*/
		numerator = numerator + (int)((denominator - numerator) * adaptive_phi);
	    }
	}
    }

    if (numfalse < lowbad){
	lowbad = numfalse;
	if (printfalse) save_false_clauses(lowbad);
	if (printlow) save_low_assign();
    }
    if (numflip >= tail_start_flip){
	tailhist[(numfalse < HISTMAX) ? numfalse : (HISTMAX - 1)] ++;
	sumfalse += numfalse;
	sumfalse_squared += numfalse * numfalse;
	sample_size ++;
    }
}

void update_and_print_statistics_end_try(void)
{
    int i;
    int j;
    double undo_fraction;

    totalflip += numflip;
    x += numflip;
    r ++;

    if (sample_size > 0){
	avgfalse = sumfalse/sample_size;
	second_moment_avgfalse = sumfalse_squared / sample_size;
	variance_avgfalse = second_moment_avgfalse - (avgfalse * avgfalse);
	if (sample_size > 1) { variance_avgfalse = (variance_avgfalse * sample_size)/(sample_size - 1); }
	std_dev_avgfalse = sqrt(variance_avgfalse);

	ratio_avgfalse = avgfalse / std_dev_avgfalse;

	sum_avgfalse += avgfalse;
	sum_std_dev_avgfalse += std_dev_avgfalse;
	number_sampled_runs += 1;

	if (numfalse <= target){
	    suc_number_sampled_runs += 1;
	    suc_sum_avgfalse += avgfalse;
	    suc_sum_std_dev_avgfalse += std_dev_avgfalse;
	}
	else {
	    nonsuc_number_sampled_runs += 1;
	    nonsuc_sum_avgfalse += avgfalse;
	    nonsuc_sum_std_dev_avgfalse += std_dev_avgfalse;
	}
    }
    else{
	avgfalse = 0;
	variance_avgfalse = 0;
	std_dev_avgfalse = 0;
	ratio_avgfalse = 0;
    }

    if(numfalse <= target){
	status_flag = 0;
	save_solution();
	numsuccesstry++;
	totalsuccessflip += numflip;
	integer_sum_x += x;
	sum_x = (double) integer_sum_x;
	sum_x_squared += ((double)x)*((double)x);
	mean_x = sum_x / numsuccesstry;
	if (numsuccesstry > 1){
	    second_moment_x = sum_x_squared / numsuccesstry;
	    variance_x = second_moment_x - (mean_x * mean_x);
	    /* Adjustment for small small sample size */
	    variance_x = (variance_x * numsuccesstry)/(numsuccesstry - 1);
	    std_dev_x = sqrt(variance_x);
	    std_error_mean_x = std_dev_x / sqrt((double)numsuccesstry);
	}
	sum_r += r;
	mean_r = ((double)sum_r)/numsuccesstry;
	sum_r_squared += ((double)r)*((double)r);
	x = 0;
	r = 0;
    }

    undo_fraction = ((double)undo_count)/numflip;
    
    printf(" %9i %9i %9.2f %9.2f %9.2f %9" BIGFORMAT " %9.6f %9i",
	   lowbad,numfalse,avgfalse, std_dev_avgfalse,ratio_avgfalse,numflip, undo_fraction, (numsuccesstry*100)/numtry);
    if (numsuccesstry > 0){
	printf(" %9" BIGFORMAT, totalsuccessflip/numsuccesstry);
	printf(" %11.2f", mean_x);
	if (numsuccesstry > 1){
	    printf(" %11.2f", std_dev_x);
	}
    }
    printf("\n");

    if (printhist){
	printf("histogram: ");
	for (i=0; i<HISTMAX; i++){
	    printf(" %" BIGFORMAT "(%i)", tailhist[i], i);
	    if ((i+1) % 10 == 0) printf("\n           ");
	}
	printf("\n");
    }

    if (numfalse>0 && printfalse)
	print_false_clauses(lowbad);
    if (printlow && (!printonlysol || numfalse >= target))
	print_low_assign(lowbad);

    if(numfalse == 0 && countunsat() != 0){
	fprintf(stderr, "Program error, verification of solution fails!\n");
	exit(-1);
    }

    fflush(stdout);
}



void print_statistics_final(void)
{
    seconds_per_flip = expertime / totalflip;
    printf("\ntotal elapsed seconds = %f\n", expertime);
    printf("average flips per second = %f\n", ((double)totalflip)/expertime);
    printf("number solutions found = %i\n", numsuccesstry);
    printf("final success rate = %f\n", ((double)numsuccesstry * 100.0)/numtry);
    printf("average length successful tries = %" BIGFORMAT "\n", numsuccesstry ? (totalsuccessflip/numsuccesstry) : 0);
    if (numsuccesstry > 0)
	{
	    printf("average flips per assign (over all runs) = %f\n", ((double)totalflip)/numsuccesstry);
	    printf("average seconds per assign (over all runs) = %f\n", (((double)totalflip)/numsuccesstry)*seconds_per_flip);
	    printf("mean flips until assign = %f\n", mean_x);
	    if (numsuccesstry>1){
		printf("  variance = %f\n", variance_x);
		printf("  standard deviation = %f\n", std_dev_x);
		printf("  standard error of mean = %f\n", std_error_mean_x);
	    }
	    printf("mean seconds until assign = %f\n", mean_x * seconds_per_flip);
	    if (numsuccesstry>1){
		printf("  variance = %f\n", variance_x * seconds_per_flip * seconds_per_flip);
		printf("  standard deviation = %f\n", std_dev_x * seconds_per_flip);
		printf("  standard error of mean = %f\n", std_error_mean_x * seconds_per_flip);
	    }
	    printf("mean restarts until assign = %f\n", mean_r);
	    if (numsuccesstry>1){
		variance_r = (sum_r_squared / numsuccesstry) - (mean_r * mean_r);
		if (numsuccesstry > 1) variance_r = (variance_r * numsuccesstry)/(numsuccesstry - 1);	   
		std_dev_r = sqrt(variance_r);
		std_error_mean_r = std_dev_r / sqrt((double)numsuccesstry);
		printf("  variance = %f\n", variance_r);
		printf("  standard deviation = %f\n", std_dev_r);
		printf("  standard error of mean = %f\n", std_error_mean_r);
	    }
	}

    if (number_sampled_runs){
      mean_avgfalse = sum_avgfalse / number_sampled_runs;
      mean_std_dev_avgfalse = sum_std_dev_avgfalse / number_sampled_runs;
      ratio_mean_avgfalse = mean_avgfalse / mean_std_dev_avgfalse;

      if (suc_number_sampled_runs){
	  suc_mean_avgfalse = suc_sum_avgfalse / suc_number_sampled_runs;
	  suc_mean_std_dev_avgfalse = suc_sum_std_dev_avgfalse / suc_number_sampled_runs;
	  suc_ratio_mean_avgfalse = suc_mean_avgfalse / suc_mean_std_dev_avgfalse;
      }
      else {
	  suc_mean_avgfalse = 0;
	  suc_mean_std_dev_avgfalse = 0;
	  suc_ratio_mean_avgfalse = 0;
      }

      if (nonsuc_number_sampled_runs){
	  nonsuc_mean_avgfalse = nonsuc_sum_avgfalse / nonsuc_number_sampled_runs;
	  nonsuc_mean_std_dev_avgfalse = nonsuc_sum_std_dev_avgfalse / nonsuc_number_sampled_runs;
	  nonsuc_ratio_mean_avgfalse = nonsuc_mean_avgfalse / nonsuc_mean_std_dev_avgfalse;
      }
      else {
	  nonsuc_mean_avgfalse = 0;
	  nonsuc_mean_std_dev_avgfalse = 0;
	  nonsuc_ratio_mean_avgfalse = 0;
      }

      printf("final numbad level statistics\n");
      printf("    statistics over all runs:\n");
      printf("      overall mean average numbad = %f\n", mean_avgfalse);
      printf("      overall mean meanbad std deviation = %f\n", mean_std_dev_avgfalse);
      printf("      overall ratio mean numbad to mean std dev = %f\n", ratio_mean_avgfalse);
      printf("    statistics on successful runs:\n");
      printf("      successful mean average numbad = %f\n", suc_mean_avgfalse);
      printf("      successful mean numbad std deviation = %f\n", suc_mean_std_dev_avgfalse);
      printf("      successful ratio mean numbad to mean std dev = %f\n", suc_ratio_mean_avgfalse);
      printf("    statistics on nonsuccessful runs:\n");
      printf("      nonsuccessful mean average numbad level = %f\n", nonsuc_mean_avgfalse);
      printf("      nonsuccessful mean numbad std deviation = %f\n", nonsuc_mean_std_dev_avgfalse);
      printf("      nonsuccessful ratio mean numbad to mean std dev = %f\n", nonsuc_ratio_mean_avgfalse);
    }

    if (hamming_flag){
	fclose(hamming_fp);
	printf("Final distance to hamming target = %i\n", calc_hamming_dist(atom, hamming_target, numatom));
	printf("Hamming distance data stored in %s\n", hamming_data_file);
    }

    if (numsuccesstry > 0){
	printf("ASSIGNMENT FOUND\n");
	if (printsolcnf == TRUE) print_sol_cnf();
	if (outfile[0]) print_sol_file(outfile);
    }
    else
	printf("ASSIGNMENT NOT FOUND\n");
}



int calc_hamming_dist(int atom[], int hamming_target[], int numatom)
{
    int i;
    int dist = 0;
    
    for (i=1; i<=numatom; i++){
	if (atom[i] != hamming_target[i]) dist++;
    }
    return dist;
}



void open_hamming_data(void)
{
    if ((hamming_fp = fopen(hamming_data_file, "w")) == NULL){
	fprintf(stderr, "Cannot open %s for output\n", initfile);
	exit(1);
    }
}



void read_hamming_file(void)
{
    int i;			/* loop counter */
    FILE * infile;
    int lit;    

    printf("loading hamming target file %s ...", hamming_target_file);

    if ((infile = fopen(hamming_target_file, "r")) == NULL){
	fprintf(stderr, "Cannot open %s\n", initfile);
	exit(1);
    }
    i=0;
    for(i = 1;i < numatom+1;i++)
      hamming_target[i] = 0;

    while (fscanf(infile, " %i", &lit)==1){
	if (ABS(lit)>numatom){
	    fprintf(stderr, "Bad hamming file %s\n", initfile);
	    exit(1);
	}
	if (lit>0) hamming_target[lit]=1;
    }
    printf("done\n");
}



void print_false_clauses(int lowbad)
{
    int i, j;
    int cl;

    printf("Unsatisfied clauses:\n");
    for (i=0; i<lowbad; i++){
	cl = lowfalse[i];
	for (j=0; j<size[cl]; j++){
	    printf("%i ", clause[cl][j]);
	}
	printf("0\n");
    }
    printf("End unsatisfied clauses\n");
}



void save_false_clauses(int lowbad)
{
    int i;

    for (i=0; i<lowbad; i++)
      lowfalse[i] = false[i];
}



void print_sol_cnf(void)
{
    int i;
    for(i = 1;i < numatom+1;i++)
	printf("v %i\n", solution[i] == 1 ? i : -i);
}



void print_sol_file(char * filename)
{
    FILE * fp;
    int i;

    if ((fp = fopen(filename, "w"))==NULL){
	fprintf(stderr, "Cannot open output file\n");
	exit(-1);
    }
    for(i = 1;i < numatom+1;i++){
	fprintf(fp, " %i", solution[i] == 1 ? i : -i);
	if (i % 10 == 0) fprintf(fp, "\n");
    }
    if ((i-1) % 10 != 0) fprintf(fp, "\n");
    fclose(fp);
}



void print_low_assign(int lowbad)
{
    int i;

    printf("Begin assign with lowest # bad = %d\n", lowbad);
    for (i=1; i<=numatom; i++){
	printf(" %i", lowatom[i]==0 ? -i : i);
	if (i % 10 == 0) printf("\n");
    }
    if ((i-1) % 10 != 0) printf("\n");
    printf("End assign\n");
}



void print_current_assign(void)
{
    int i;

    printf("Begin assign at flip = %" BIGFORMAT "\n", numflip);
    for (i=1; i<=numatom; i++){
	printf(" %i", atom[i]==0 ? -i : i);
	if (i % 10 == 0) printf("\n");
    }
    if ((i-1) % 10 != 0) printf("\n");
    printf("End assign\n");
}



void save_low_assign(void)
{
    int i;

    for (i=1; i<=numatom; i++)
      lowatom[i] = atom[i];
}



void save_solution(void)
{
    int i;

    for (i=1; i<=numatom; i++)
      solution[i] = atom[i];
}



/*******************************************************/
/* Utility Functions                                   */
/*******************************************************/


long super(int i)
{
    long power;
    int k;

    if (i<=0){
	fprintf(stderr, "bad argument super(%i)\n", i);
	exit(1);
    }
    /* let 2^k be the least power of 2 >= (i+1) */
    k = 1;
    power = 2;
    while (power < (i+1)){
	k += 1;
	power *= 2;
    }
    if (power == (i+1)) return (power/2);
    return (super(i - (power/2) + 1));
}



void handle_interrupt(int sig)
{
    if (abort_flag) exit(-1);
    abort_flag = TRUE;
}



void scanone(int argc, char *argv[], int i, int *varptr)
{
    BIGINT n;
    scanonell(argc, argv, i, &n);
    *varptr = (int) n;
}



void scanoned(int argc, char *argv[], int i, double *varptr)
{
    if (i>=argc || sscanf(argv[i],"%lf",varptr)!=1){
	fprintf(stderr, "Bad argument %s\n", i<argc ? argv[i] : argv[argc-1]);
	exit(-1);
    }
}


void scanonell(int argc, char *argv[], int i, BIGINT *varptr)
{
    char buf[25];
    int factor = 1;
    BIGINT n;

    if (i>=argc || strlen(argv[i])>24){
	fprintf(stderr, "Bad argument %s\n", i<argc ? argv[i] : argv[argc-1]);
	exit(-1);
    }
    strcpy(buf, argv[i]);
    switch (buf[strlen(buf)-1]){
    case 'K':
	factor = 1000;
	buf[strlen(buf)-1] = 0;
	break;
    case 'M':
	factor = 1000000;
	buf[strlen(buf)-1] = 0;
	break;
    case 'B':
	factor = 1000000000;
	buf[strlen(buf)-1] = 0;
	break;
    }
    if (sscanf(argv[i],"%" BIGFORMAT , &n)!=1){
	fprintf(stderr, "Bad argument %s\n", i<argc ? argv[i] : argv[argc-1]);
	exit(-1);
    }
    n = n * factor;
    *varptr = n;
}


int countunsat(void)
{
    int i, j, unsat, bad, lit, sign;

    unsat = 0;
    for (i=0;i < numclause;i++) {
	bad = TRUE;
	for (j=0; j < size[i]; j++) {
	    lit = clause[i][j];
	    sign = lit > 0 ? 1 : 0;
	    if ( atom[ABS(lit)] == sign )
	    {
		bad = FALSE;
		break;
	    }
	}
	if (bad)
	    unsat++;
    }
    return unsat;
}



double elapsed_seconds(void) 
{ 
	double answer;

#if POSIX
	static long prev_time = 0;
	time( &long_time );
	/* Note:  time(&t) returns t in seconds, so do not need to /CLK_TCK */
	answer = long_time - prev_time;
	prev_time = long_time;
#elif WINDOWS
	static DWORD prev_time = 0;
	win_time = timeGetTime();
	/* Note:  return value of timeGetTime() is ms, so divide by 1000*/
	answer = (double)(win_time - prev_time) / (double)1000; 
	prev_time = win_time;
#elif BSD || LINUX || OSX
    static struct tms prog_tms;
    static long prev_times = 0;
    (void) times(&prog_tms);
    answer = ((double)(((long)prog_tms.tms_utime)-prev_times))/((double) ticks_per_second);
    prev_times = (long) prog_tms.tms_utime;
#endif
	return answer; 
}



/****************************************************************/
/*                  Heuristics                                  */
/****************************************************************/



int pickrandom(void)
{
    int tofix;

    tofix = false[RANDMOD(numfalse)];
    return ABS(clause[tofix][RANDMOD(tofix)]);
}



int pickbest(void)
{
    int numbreak;
    int tofix;
    int clausesize;
    int i;		
    register int numbest;
    register int bestvalue;
    register int var;

    tofix = false[ RANDMOD(numfalse) ];
    clausesize = size[tofix];
    numbest = 0;
    bestvalue = BIG;

    for (i=0; i< clausesize; i++){
      var = ABS(clause[tofix][i]);
      numbreak = breakcount[var];
      if (numbreak<=bestvalue){
	if (numbreak<bestvalue) numbest=0;
	bestvalue = numbreak;
	best[numbest++] = var;
      }
    }

    if ((nofreebie || bestvalue>0) && (RANDMOD(denominator) < numerator))
	return ABS(clause[tofix][ RANDMOD(clausesize) ]);
    return ABS(best[RANDMOD(numbest)]);
}



int pickgsat(void)
{
    int delta;
    int tofix;
    int clausesize;
    int i;		
    register int numbest;
    register int bestvalue;
    register int var;

    tofix = false[ RANDMOD(numfalse) ];
    clausesize = size[tofix];
    numbest = 0;
    bestvalue = BIG;

    for (i=0; i< clausesize; i++){
      var = ABS(clause[tofix][i]);
      delta = breakcount[var] - makecount[var];
      if (delta<=bestvalue){
	if (delta<bestvalue) numbest=0;
	bestvalue = delta;
	best[numbest++] = var;
      }
    }

    if (RANDMOD(denominator) < numerator)
	return ABS(clause[tofix][ RANDMOD(clausesize) ]);
    return ABS(best[RANDMOD(numbest)]);
}




int pickalternate(void)
{
    int numbreak;
    int tofix;
    int clausesize;
    int i;		
    register int numbest;
    register int bestvalue;
    register int var;

    if (alternate_run_remaining == 0){
	alternate_greedy_state = ! alternate_greedy_state;
	alternate_run_remaining =
	    alternate_greedy_state ? alternate_greedy : alternate_walk;
    }

    tofix = false[ RANDMOD(numfalse) ];
    clausesize = size[tofix];
    numbest = 0;
    bestvalue = BIG;

    for (i=0; i< clausesize; i++){
	var = ABS(clause[tofix][i]);
	numbreak = breakcount[var];
	if (numbreak<=bestvalue){
	    if (numbreak<bestvalue) numbest=0;
	    bestvalue = numbreak;
	    best[numbest++] = var;
	}
    }

    alternate_run_remaining --;

    if ((nofreebie || bestvalue>0) && ! alternate_greedy_state)
	return ABS(clause[tofix][ RANDMOD(clausesize) ]);

    if (bestvalue>0 && numerator > 0 && RANDMOD(denominator) <= numerator)
	return ABS(clause[tofix][ RANDMOD(clausesize) ]);
    
    return ABS(best[RANDMOD(numbest)]);
}



int pickbigflip(void)
{
    int numbreak;
    int tofix;
    int clausesize;
    int i;		
    register int numbest;
    register int bestvalue;
    register int var;

    if (alternate_run_remaining == 0){
	if (RANDMOD(denominator) < numerator){
	    alternate_greedy_state = FALSE;
	    alternate_run_remaining = alternate_walk;
	}
	else {
	    alternate_greedy_state = TRUE;
	    alternate_run_remaining = alternate_greedy;
	}
    }

    tofix = false[ RANDMOD(numfalse) ];
    clausesize = size[tofix];
    numbest = 0;
    bestvalue = BIG;

    for (i=0; i< clausesize; i++){
	var = ABS(clause[tofix][i]);
	numbreak = breakcount[var];
	if (numbreak<=bestvalue){
	    if (numbreak<bestvalue) numbest=0;
	    bestvalue = numbreak;
	    best[numbest++] = var;
	}
    }

    alternate_run_remaining --;

    if ((nofreebie || bestvalue>0) && ! alternate_greedy_state)
	return ABS(clause[tofix][ RANDMOD(clausesize) ]);
    return ABS(best[RANDMOD(numbest)]);
}



/* References for versions of Novelty:

   novelty and r-novelty:
   Evidence for Invariants in Local Search. David McAllester, Bart Selman, and Henry Kautz.

   novelty+ and r-novelty+:
   On the Run-time Behaviour of Stochastic Local Search Algorithms for SAT. Holger H. Hoos.

   adaptive novelty+:
   An Adaptive Noise Mechanism for WalkSAT (Corrected). Holger H. Hoos.
*/



int pickrnovelty(void)
{
  int var, diff, birthdate;
  int diffdiff;
  int youngest, youngest_birthdate, best, second_best, best_diff, second_best_diff;
  int tofix, clausesize, i;

  tofix = false[RANDMOD(numfalse)];
  clausesize = size[tofix];  

  if (clausesize == 1) return ABS(clause[tofix][0]);

  if (plus_flag){
      if (RANDMOD(denominator) <= ONE_PERCENT) return ABS(clause[tofix][RANDMOD(clausesize)]);
  }
  else {
      if ((numflip % 100) == 0) return ABS(clause[tofix][RANDMOD(clausesize)]);
  }

  youngest_birthdate = -BIG;
  best_diff = -BIG;
  second_best_diff = -BIG;

  for(i = 0; i < clausesize; i++){
      var = ABS(clause[tofix][i]);
      diff = makecount[var] - breakcount[var];
      birthdate = changed[var];
      if (birthdate > youngest_birthdate){
	  youngest_birthdate = birthdate;
	  youngest = var;
      }
      if (diff > best_diff || (diff == best_diff && changed[var] < changed[best])) {
	  /* found new best, demote best to 2nd best */
	  second_best = best;
	  second_best_diff = best_diff;
	  best = var;
	  best_diff = diff;
      }
      else if (diff > second_best_diff || (diff == second_best_diff && changed[var] < changed[second_best])){
	  /* found new second best */
	  second_best = var;
	  second_best_diff = diff;
      }
  }
  if (best != youngest) return best;

  diffdiff = best_diff - second_best_diff;

  /* If best is youngest, then second best must be strictly worse */
  if (diffdiff <= 0){
      fprintf(stderr, "rnovelty+ code error!\n");
      fprintf(stderr, "diffdiff = %i\n", diffdiff);
      fprintf(stderr, "best = %i   best_diff = %i   second_best = %i    second_best_diff = %i\n", best, best_diff, second_best, second_best_diff);
      exit(-1);
  }

  /* (1) p < 0.5 and n > 1 */
  if (numerator*2 < denominator && diffdiff > 1) return best;

  /* (2) p < 0.5 and n = 1                                 */
  /*     with probability 2p pick 2nd best, otherwise best */
  if (numerator*2 < denominator && diffdiff == 1){
    if ((RANDMOD(denominator)) < 2*numerator) return second_best;
    return best;
  }

  /* (3) p >= 0.5 and n = 1 */
  if (diffdiff == 1) return second_best;

  /* (4) p >= 0.5 and n > 1 (only remaining case)                   */
  /*     with probability 2(p-0.5) pick second best, otherwise best */

  if ((RANDMOD(denominator)) < 2*(numerator-(denominator/2))) return second_best;

  return best;

}



int picknovelty(void)
{
  int var, diff, birthdate;
  int diffdiff;
  int youngest, youngest_birthdate, best, second_best, best_diff, second_best_diff;
  int tofix, clausesize, i;

  tofix = false[RANDMOD(numfalse)];
  clausesize = size[tofix];  

  if (clausesize == 1) return ABS(clause[tofix][0]);

  if (plus_flag){
      if (RANDMOD(denominator) <= ONE_PERCENT) return ABS(clause[tofix][RANDMOD(clausesize)]);
  }
  else {
      if ((numflip % 100) == 0) return ABS(clause[tofix][RANDMOD(clausesize)]);
  }

  youngest_birthdate = -BIG;
  best_diff = -BIG;
  second_best_diff = -BIG;

  for(i = 0; i < clausesize; i++){
    var = ABS(clause[tofix][i]);
    diff = makecount[var] - breakcount[var];
    birthdate = changed[var];
    if (birthdate > youngest_birthdate){
      youngest_birthdate = birthdate;
      youngest = var;
    }
    if (diff > best_diff || (diff == best_diff && changed[var] < changed[best])) {
      /* found new best, demote best to 2nd best */
      second_best = best;
      second_best_diff = best_diff;
      best = var;
      best_diff = diff;
    }
    else if (diff > second_best_diff || (diff == second_best_diff && changed[var] < changed[second_best])){
      /* found new second best */
      second_best = var;
      second_best_diff = diff;
    }
  }

  /* Case 1: If best variable not youngest, select it */
  if (best != youngest) return best;

  /* Case 2: with probability p select second best, else best */
  if ((RANDMOD(denominator)) <= numerator) return second_best;
  return best;
}



int picktabu(void)
{
    int numbreak;
    int tofix;
    int clausesize;
    int i;			/* a loop counter */
    int numbest;		/* how many are tied for best */
    int numbesttabu;
    int numany;
    int bestvalue;
    int besttabuvalue;
    int attempt;
    int var;

    for (attempt=0; attempt < MAXATTEMPT; attempt++){
    
	tofix = false[RANDMOD(numfalse)];
	clausesize = size[tofix];
	numbest = 0;
	numbesttabu = 0;
	bestvalue = BIG;
	besttabuvalue = BIG;
	numany = 0;

	for (i=0; i< clausesize; i++){
	    var = ABS(clause[tofix][i]);
	    numbreak = breakcount[var];
	    if (numbreak<=besttabuvalue && (tabu_length < numflip - changed[var])){
		if (numbreak<besttabuvalue) numbesttabu=0;
		besttabuvalue = numbreak;
		besttabu[numbesttabu++] = var;
	    }
	    if (numbreak<=bestvalue){
		if (numbreak<bestvalue) numbest=0;
		bestvalue = numbreak;
		best[numbest++] = var;
	    }
	    if (tabu_length < numflip - changed[var]) {
		any[numany++] = var;
	    }
	}

	/* Handle freebie picks */
	if (bestvalue == 0 && ! nofreebie){
	    if (numbesttabu > 0) return besttabu[RANDMOD(numbesttabu)];
	    return best[RANDMOD(numbest)];
	}
	if (numerator > 0 && RANDMOD(denominator) < numerator){
	    /* Handle random walk picks */
	    if (numany > 0) return any[RANDMOD(numany)];
	    if (attempt == MAXATTEMPT - 1) return ABS(clause[tofix][RANDMOD(clausesize)]);
	}
	else { /* Handle greedy picks */
	    if (numbesttabu > 0) return besttabu[RANDMOD(numbesttabu)];
	    if (attempt == MAXATTEMPT - 1) return best[RANDMOD(numbest)];
	}
    }
    fprintf(stderr, "Bug in tabu!\n");
    exit(-1);
}
