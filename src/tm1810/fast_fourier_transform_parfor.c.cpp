#include "fourier_transform.hpp"
#include "tbb/parallel_for.h"
#include <cmath>
#include <cassert>
#include <iostream>

namespace hpce
{
	namespace tm1810
	{
		class fast_fourier_transform_parfor
			: public fourier_transform
		{
		protected:
			/* Standard radix-2 FFT only supports binary power lengths */
			virtual size_t calc_padded_size(size_t n) const
			{
				assert(n!=0);
				
				size_t ret=1;
				while(ret<n){
					ret<<=1;
				}
				
				return ret;
			}

			virtual void forwards_impl(
				size_t n,	const std::complex<double> &wn,
				const std::complex<double> *pIn, size_t sIn,
				std::complex<double> *pOut, size_t sOut
			) const 
			{
				assert(n>0);
				
				if (n == 1){
					pOut[0] = pIn[0];
				}else if (n == 2){
					pOut[0] = pIn[0]+pIn[sIn];
					pOut[sOut] = pIn[0]-pIn[sIn];
				}else{
					size_t m = n/2;

					forwards_impl(m,wn*wn,pIn,2*sIn,pOut,sOut);
					forwards_impl(m,wn*wn,pIn+sIn,2*sIn,pOut+sOut*m,sOut);
					 
					
					
					
					std::complex<double> w65=std::complex<double>(1.0, 0.0);//might need to move it afgain
					std::complex<double> w=std::complex<double>(1.0, 0.0);//might need to move it afgain
					
					size_t K = 8;
					size_t acc=0;
					size_t acctemp=0;
					tbb::parallel_for((size_t) 0, (m/K), [=](size_t j0){
					std::complex<double>  wparfor = w * (std::complex<double>)std::pow(wn,j0*K);//calculate base W for each sequential chunk
					//std::cerr << "Wparfor is " << wparfor << "\r\n";
						for (size_t j1=0; j1<K; j1++){
						  size_t j=j0*K+j1;
						  std::complex<double> t1 = wparfor*pOut[m+j];
						  std::complex<double> t2 = pOut[j]-t1;
						  pOut[j] = pOut[j]+t1;                 /*  pOut[j] = pOut[j] + w^i pOut[m+j] */
						  pOut[j+m] = t2;                          /*  pOut[j] = pOut[j] - w^i pOut[m+j] */
						  wparfor = wparfor*wn;
						} 
					} );
				if (m%K) {//edge case where the recursion doesnt know how to handle m if its < K. 
					//std::cerr << "m is " << m << " mod left over is " << m%K << std::endl;
					return;
				}
				else{
						//Edge case to deal with non divisble m/k
						for (size_t j0=m/K; j0<(m/K)+1; j0++){
							std::complex<double>  wedge= w * (std::complex<double>)std::pow(wn,j0*K);
							//std::cerr << "EDGE CASE "<< "\r\n";
							for (size_t j1=0; j1<(m); j1++){//do the last few iterations(mod calcs the remainder left to do).
							  size_t j=j0*K+j1;
							  std::complex<double> t1 = wedge*pOut[m+j];
							  std::complex<double> t2 = pOut[j]-t1;
							  pOut[j] = pOut[j]+t1;                 
							  pOut[j+m] = t2;                          
							  wedge = wedge*wn;
							}
						}
					}

				}
			}
			
			virtual void backwards_impl(
				size_t n,	const std::complex<double> &wn,
				const std::complex<double> *pIn, size_t sIn,
				std::complex<double> *pOut, size_t sOut
			) const 
			{
				complex_t reverse_wn=1.0/wn;
				forwards_impl(n, reverse_wn, pIn, sIn, pOut, sOut);
				
				double scale=1.0/n;
				for(size_t i=0;i<n;i++){
					pOut[i]=pOut[i]*scale;
				}
			}
			
		public:
			virtual std::string name() const
			{ return "hpce.ch3810.fast_fourier_transform_parfor"; }
			
			virtual bool is_quadratic() const
			{ return false; }
		};

		std::shared_ptr<fourier_transform> Create_fast_fourier_transform_parfor()
		{
			return std::make_shared<fast_fourier_transform_parfor>();
		}
	};//namespace ch3810
}; // namespace hpce
