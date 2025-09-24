Walksat - Version 56
https://gitlab.com/HenryKautz/Walksat
May 2018
Please email bug reports to
Henry Kautz <henry.kautz@gmail.com>

Compilation

Just type:
        make 

Compilation options:
	make install   # compile and copy to $USER/bin
	make clean     # delete binaries

For Windows Visual Studio, create a workspace/project and add the
library Winmm.lib to the "linker" tab.

/* Platform dependent differences:                                  */
/*    -arc4 only available on BSD and OSX. Option uses slower but   */
/*        stronger random number generation arc4random and          */
/*        arc4random_uniform                                        */
/*    WINDOWS and POSIX use rand() instead of random()              */
/*    Clock ticks per second determined by sysconf(_SC_CLK_TCK)     */
/*        for BSD, OSX, and LINUX                                   */
/*    Clock ticks per second fixed at 1000 for Windows              */
/*    Clock ticks per second fixed at 1 for POSIX                   */

Walksat Specification

Walksat attempts to find a satisfying model of a conjective normal
form formula.  Input is cnf format:

c Optional comments at start of file.
c The "p" line specifies cnf format, number vars, number clauses
c Variables are numbered starting at 1
c Each clause ends with 0
p cnf 3 2
1 -3 0
2 3 -1 0

Walksat reads from standard in or from a file specified as a command
line argument and writes to standard out and standard error.  To
get a list of command line options, type
	walksat -help
or any illegal option.

*****************************************************************

Also included in distribution:

makewff [-seed N] [-cnf] [-f] [-kf] clen nvars nclauses
Generate random formulas using the fixed clause length model.
The -seed argument can be used to initialize the random number
generator.  The -cnf, -f, and -kf options specify the format of the
wff file; for walksat use -cnf.  Clen is the length of each clause,
nvars the number of variables, and nclauses the number of clauses.

makequeens N
Generate a N-queens formula.

generate-wffs.sh
Generate a numbered sequence of random 3-CNF formulas.
Use: generate-wffs.sh number_vars ratio start_wff_numer end_wff_number
Enter ratio as a percentage, e.g. 425 == 4.25 clause/variable.

generate-hard-wffs.sh
Use: generate-hard-wffs.sh number_vars start_wff_numer end_wff_number
Same as generate-wffs.sh, but with ratio fixed at 4.25.

*****************************************************************
Walksat:
Bart Selman, Henry A. Kautz, and Bram Cohen. Noise Strategies for Improving Local Search. Proceedings AAAI-94, Seattle, WA, 1994.

Novelty heuristics:
David McAllester, Bart Selman, and Henry Kautz. Evidence for Invariants in Local Search. Proceedings AAAI-97, Providence, RI, 1997.

Plus option for novelty heuristics:
Holger H. Hoos, On the Run-time Behaviour of Stochastic Local Search Algorithms for SAT. Proceedings AAAI-99, Orlando, FL, 1999.

Adaptive cutoff option:
Holger H. Hoos, An Adaptive Noise Mechanism for WalkSAT. Proceedings AAAI-2002, Edmonton, Alberta, Canada, 2002.

*****************************************************************

HISTORY OF MAJOR RELEASES

version 07 - introduced breakcount[] array
version 14 - added -super option
version 16 - improved print options and help message
version 17 - added trace N option
version 25 - -hamming option
version 28 - novelty and rnovelty heuristics
version 41 - use long long ints for cutoff
version 44 - special version for sat competition - not distributed
version 45 - watched literals rule added
version 48 - allow cnf file name to appear as argument on command line
version 49 - various minor bugs, watched literals rule removed
version 50 - code optimizations suggested by Don Knuth
version 53 - added -plus option for -novelty and -rnovelty
             fixed -tabu;
	     added heuristic -alternate;
	     added heuristic -bigflip;
	     added -adaptive noise option;
	     added -nofreebie option;
	     added -maxfreebie option;
	     added -walkprob R to substitute for -noise N M option;
	     replaced macro functions with inline functions;
	     added directory of Papers to distribution;
	     allow K and M as suffixes for -cutoff;
	     moved distribution to gitlab.com
version 54 - allocates all data structures dynamically
version 55 - include -arc4 to use arc4random random number generator.
version 56 - updated preprocessor #defines
