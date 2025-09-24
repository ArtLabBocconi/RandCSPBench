#include <stdio.h>
#include <stdlib.h>

int n;

int generate(int p)
{
    int i,j,k;
    int c;
    c =0;
    /* queen on every row */
    for (j=1; j<=n; j++){
        for (i=1; i<=n; i++){
            if (p) printf("%i ", n*(j-1)+i);
        }
        c++; if (p) printf(" 0\n");
    }
    /* no horizontal attacks */
    for (j=1; j<=n; j++){
        for (i=1; i<n; i++){
            for (k=i+1; k<=n; k++){
                c++; if (p) printf("-%i -%i 0\n", n*(j-1)+i, n*(j-1)+k);
            }
        }
    }
    /* no vertical attacks */
    for (i=1; i<=n; i++){
        for (j=1; j<n; j++){
            for (k=j+1; k<=n; k++){
                c++; if (p) printf("-%i -%i 0\n", n*(j-1)+i, n*(k-1)+i);
            }
        }
    }
    /* no SE attacks */
    for (i=1; i<n; i++){
        for (j=1; j<n; j++){
            for (k=1; (i+k)<=n && (j+k)<=n; k++){
                c++; if (p) printf("-%i -%i 0\n", n*(j-1)+i, n*(j-1+k)+i+k);
            }
        }
    }
    /* no SW attacks */
    for (i=2; i<=n; i++){
        for (j=1; j<n; j++){
            for (k=1; (i-k)>=1 && (j+k)<=n; k++){
                c++; if (p) printf("-%i -%i 0\n", n*(j-1)+i, n*(j-1+k)+i-k);
            }
        }
    }
    return c;
}

int main(int argc, char ** argv)
{
    if (argc < 2) return(1);
    n = atoi(argv[1]);

    printf("p cnf %i %i\n", n*n, generate(0));
    generate(1);

    return(0);
}





