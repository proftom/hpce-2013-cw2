#include "fourier_transform.hpp"

#include <cmath>
#include <cassert>
#include "tbb/task_group.h"
#include "tbb/parallel_for.h"

#define problemSizeToParallise 4096
#define K 4096
namespace hpce
{
	namespace tm1810 {
		class fast_fourier_transform_opt
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
				if(n==16) {
					forwards_impl_unroll(16,wn,pIn,sIn,pOut,sOut);
				}
				if (n == 1){
					pOut[0] = pIn[0];
				}else if (n == 2){
					pOut[0] = pIn[0]+pIn[sIn];
					pOut[1] = pIn[0]-pIn[sIn];
				} else {
				
					size_t m = n/2;
					//Problem has to be a decent size to amortise scheduling cost
					if (m > problemSizeToParallise) {
						tbb::task_group group;
						group.run([=]() {forwards_impl(m,wn*wn,pIn,2*sIn,pOut,sOut); });
						group.run([=]()	{forwards_impl(m,wn*wn,pIn+sIn,2*sIn,pOut+sOut*m,sOut);});
						group.wait();
					} else {
						forwards_impl(m,wn*wn,pIn,2*sIn,pOut,sOut);
						forwards_impl(m,wn*wn,pIn+sIn,2*sIn,pOut+sOut*m,sOut);
					}
					

						//If the problem is too little to warrant parallel execution
                        if (m <= K){
                                std::complex<double> w = std::complex<double>(1.0, 0.0);
                                for (size_t j = 0; j<m; j++){
                                    std::complex<double> t1 = w*pOut[m+j];
									std::complex<double> t2 = pOut[j]-t1;
									pOut[j] = pOut[j]+t1;                 
									pOut[j+m] = t2;                       
									w = w*wn;
                                }
                        } else {
							tbb::parallel_for(size_t(0), m/K, [=](size_t j0){
									std::complex<double> w = std::pow(wn, j0*K);
									for (size_t j1 = j0*K; j1 < (j0+1)*K; j1++){
										std::complex<double> t1 = w*pOut[m+j1];
										std::complex<double> t2 = pOut[j1]-t1;
										pOut[j1] = pOut[j1]+t1;                 
										pOut[j1+m] = t2;                       
										w = w*wn;
									}
							}
							);
                        }
			
				}
			}
			

        static inline void forwards_impl_unroll(

                size_t n, const std::complex<double> &wn,
                const std::complex<double> *pIn, size_t sIn,
                std::complex<double> *pOut, size_t sOut
        )
			{
			
				//assert(n>0);
				
				if (n == 2){
					pOut[0] = pIn[0]+pIn[sIn];
					pOut[1] = pIn[0]-pIn[sIn];
				} else{
					size_t m = n/2;

					forwards_impl_unroll(m,wn*wn,pIn,2*sIn,pOut,sOut);
					forwards_impl_unroll(m,wn*wn,pIn+sIn,2*sIn,pOut+sOut*m,sOut);

					std::complex<double>  w = std::complex<double>(1.0, 0.0);
					for (size_t j=0;j<m;j++){
					  std::complex<double> t1 = w*pOut[m+j];
					  std::complex<double> t2 = pOut[j]-t1;
					  pOut[j] = pOut[j]+t1;
					  pOut[j+m] = t2;
					  w = w*wn;
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
			{ return "hpce.tm1810.fast_fourier_transform_opt"; }
			
			virtual bool is_quadratic() const
			{ return false; }
		};

		std::shared_ptr<fourier_transform> Create_fast_fourier_transform_opt()
		{
			return std::make_shared<fast_fourier_transform_opt>();
		}
	}
}; // namespace hpce
