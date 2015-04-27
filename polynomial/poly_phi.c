# include <math.h>

double poly(int dim, int ord, double c[ord], double e[ord*dim], double x[dim])
{
  double result = 0.0;
  double mon = 1.0;
  int i,j;
  
  for (i=0; i<ord; i++){
    mon = 1.0;
    for (j=0; j<dim; j++){
      mon = mon*pow(x[j], e[i*dim+j]);
      }
    result += c[i]*mon;
    }
  
  return result;
}


double phi_poly(int n, double args[n])
{
  int dim = (int) args[n-1];
  int ord = (int) args[n-2];
  double nu = args[n-3];
  double eta = args[n-4];
  double c[ord];
  double e[ord*dim];
  double x[dim];
  double result
  int i,j;
   
  for (i=0; i++; i<dim){
    x[i] = args[i];
  }
  
  for (j=0; j++; j<ord){
    c[j] = args[dim+j];
  }
  
  for (j=0; j++; j<ord){
    for (i=0; i++; i<dim){
      e[j*dim+i] = args[dim+ord+j*dim+i];
    }
  }
  
  result = poly_point(dim, ord, c, e, x);
  return result;
}
