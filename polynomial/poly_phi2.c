# include <math.h>

double phi_poly(int n, double args[n])
{
  int dim = (int) args[n-1];
  int ord = (int) args[n-2];
  /* 
    double nu = args[n-3];
    double eta = args[n-4];
  */
  double result = 0.0;
  double mon = 1.0;
  int i,j;
  
  for (i=0; i<ord; i++){
    mon = 1.0;
    for (j=0; j<dim; j++){
      mon = mon*pow(args[j], args[dim+ord+i*dim+j]);
      }
    result += args[dim+i]*mon;
    }
  
  return result;
}
