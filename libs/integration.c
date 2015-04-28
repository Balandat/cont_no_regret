# include <math.h>

/* 
    Notation:
		n:    Dimension of the ambient space (i.e. s in R^n)
		m:    "order" or the polynomial (i.e. number of distinct monomials)
		c:    coefficients of the monomials (c_1,...,c_m)
		e:    exponents of the monomials (e_11,...,e_1n,e_21,...,e_2n,......,e_m1,...e_mn)
		eta:  learning rate
		nu:   shifting parameter
*/

double poly(int dim, int ord, double c[ord], double e[ord*dim], double s[dim])
{ 
/* Basic function to evaluate polynomial. */
  double result = 0.0;
  double mon = 1.0;
  int i,j;
  
  for (i=0; i<ord; i++){
    mon = 1.0;
    for (j=0; j<dim; j++){
      mon = mon*pow(s[j], e[i*dim+j]);
      }
    result += c[i]*mon;
    }
  return result;
}

double composite_potential(double u, double gamma)
{ 
/* Composite 0-potential. */
  double c = 1.0/(gamma - 1.0);
  double ctilde = gamma*c;
  double a0, a1, a2;
   
   if(u<c){
     return pow(ctilde - u, -gamma);
     }
   else{
     a2 = 0.5*gamma*(1.0+gamma);
     a1 = gamma - 2*c*a2;
     a0 = 1 - c*a1 - pow(c, 2)*a2;
     return a0 + a1*u + a2*pow(u, 2);
     }
}

double poly_ctypes(int N, double args[N])
{ 
/* Provides interface of poly to the nquad ctypes calling convention:
		args[0] - args[n-1]         : (s_1,...,s_n)
		args[n] - args[n+m-1]       : (c_1,...,c_m)
		args[n+m] - args[n+m+n*m-1] : (e_11,...,e_1n,e_21,...,e_2n,......,e_m1,...e_mn)
		args[n+m+n*m]				: m
		args[n+m+n*m+1]				: n
*/  
  int n = (int) args[N-1];
  int m = (int) args[N-2];
  
  return poly(n, m, &args[n], &args[n+m], &args[0]);
}

double exp_poly(int N, double args[N])
{ 
/* Exponential potential with polynomial losses: 
		args[0] - args[n-1]         : (s_1,...,s_n)
		args[n] - args[n+m-1]       : (c_1,...,c_m)
		args[n+m] - args[n+m+n*m-1] : (e_11,...,e_1n,e_21,...,e_2n,......,e_m1,...e_mn)
		args[n+m+n*m]				: m
		args[n+m+n*m+1]				: n
		args[n+m+n*m+2]				: eta
		args[n+m+n*m+3]				: nu
*/  
  return exp(-args[N-2]*(poly_ctypes(N-2, &args[0]) + args[N-1]));
}

double identity_poly(int N, double args[N])
{ 
/* Identity potential with polynomial losses
		args[0] - args[n-1]         : (s_1,...,s_n)
		args[n] - args[n+m-1]       : (c_1,...,c_m)
		args[n+m] - args[n+m+n*m-1] : (e_11,...,e_1n,e_21,...,e_2n,......,e_m1,...e_mn)
		args[n+m+n*m]				: m
		args[n+m+n*m+1]				: n
		args[n+m+n*m+2]				: eta
		args[n+m+n*m+3]				: nu
*/  
  return -args[N-2]*(poly_ctypes(N-2, &args[0]) + args[N-1]);
}

double composite_poly(int N, double args[N])
{ 
/* Composite 0-potential with polynomial losses:
		args[0] - args[n-1]         : (s_1,...,s_n)
		args[n] - args[n+m-1]       : (c_1,...,c_m)
		args[n+m] - args[n+m+n*m-1] : (e_11,...,e_1n,e_21,...,e_2n,......,e_m1,...e_mn)
		args[n+m+n*m]				: m
		args[n+m+n*m+1]				: n
		args[n+m+n*m+2]				: eta
		args[n+m+n*m+3]				: nu
		args[n+m+n*m+4]				: gamma
*/  
  return composite_potential(-args[N-3]*(poly_ctypes(N-3, &args[0]) + args[N-2]), args[N-1]);
}
