#include <iostream>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */
#include <cstdint>
#include <assert.h>
#include <vector>       // std::vector
#include <algorithm>    // std::random_shuffle
#include <random>
#include <stdexcept>
#include<bits/stdc++.h>

using namespace std;

using idx_t = std::uint32_t;
using val_t = float;
using ptr_t = std::uintptr_t;

/**
 * CSR structure to store search results
 */
typedef struct csr_t {
  idx_t nrows; // number of rows
  idx_t ncols; // number of rows
  idx_t * ind; // column ids
  val_t * val; // values
  ptr_t * ptr; // pointers (start of row in ind/val)

  csr_t()
  {
    nrows = ncols = 0;
    ind = nullptr;
    val = nullptr;
    ptr = nullptr;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param nnz   Number of non-zeros
   */
  void reserve(const idx_t nrows, const ptr_t nnz)
  {
    if(nrows > this->nrows){
      if(ptr){
        ptr = (ptr_t*) realloc(ptr, sizeof(ptr_t) * (nrows+1));
      } else {
        ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
        ptr[0] = 0;
      }
      if(!ptr){
        throw std::runtime_error("Could not allocate ptr array.");
      }
    }
    if(nnz > ptr[this->nrows]){
      if(ind){
        ind = (idx_t*) realloc(ind, sizeof(idx_t) * nnz);
      } else {
        ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
      }
      if(!ind){
        throw std::runtime_error("Could not allocate ind array.");
      }
      if(val){
        val = (val_t*) realloc(val, sizeof(val_t) * nnz);
      } else {
        val = (val_t*) malloc(sizeof(val_t) * nnz);
      }
      if(!val){
        throw std::runtime_error("Could not allocate val array.");
      }
    }
    this->nrows = nrows;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param factor   Sparsity factor
   */
  static csr_t * random(const idx_t nrows, const idx_t ncols, const double factor)
  {
    ptr_t nnz = (ptr_t) (factor * nrows * ncols);
    if(nnz >= nrows * ncols / 2.0){
      throw std::runtime_error("Asking for too many non-zeros. Matrix is not sparse.");
    }
    auto mat = new csr_t();
    mat->reserve(nrows, nnz);
    mat->ncols = ncols;

    /* fill in ptr array; generate random row sizes */
    unsigned int seed = (unsigned long) mat;
    long double sum = 0;
    for(idx_t i=1; i <= mat->nrows; ++i){
      mat->ptr[i] = rand_r(&seed) % ncols;
      sum += mat->ptr[i];
    }
    for(idx_t i=0; i < mat->nrows; ++i){
      double percent = mat->ptr[i+1] / sum;
      mat->ptr[i+1] = mat->ptr[i] + (ptr_t)(percent * nnz);
      if(mat->ptr[i+1] > nnz){
        mat->ptr[i+1] = nnz;
      }
    }
    if(nnz - mat->ptr[mat->nrows-1] <= ncols){
      mat->ptr[mat->nrows] = nnz;
    }

    /* fill in indices and values with random numbers */
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned long) mat * (1+tid);
      std::vector<int> perm;
      for(idx_t i=0; i < ncols; ++i){
        perm.push_back(i);
      }
      std::random_device seeder;
      std::mt19937 engine(seeder());

      #pragma omp for
      for(idx_t i=0; i < nrows; ++i){
        std::shuffle(perm.begin(), perm.end(), engine);
        for(ptr_t j=mat->ptr[i]; j < mat->ptr[i+1]; ++j){
          mat->ind[j] = perm[j - mat->ptr[i]];
          mat->val[j] = ((double) rand_r(&seed)/rand_r(&seed));
        }
      }
    }

    return mat;
  }

  string info(const string name="") const
  {
    return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
      (ptr ? to_string(ptr[nrows]) : "0") + ">";
  }

  ~csr_t()
  {
    if(ind){
      free(ind);
    }
    if(val){
      free(val);
    }
    if(ptr){
      free(ptr);
    }
  }
} csr_t;

/**
 * Ensure the matrix is valid
 * @param mat Matrix to test
 */
void test_matrix(csr_t * mat){
  auto nrows = mat->nrows;
  auto ncols = mat->ncols;
  assert(mat->ptr);
  auto nnz = mat->ptr[nrows];
  for(idx_t i=0; i < nrows; ++i){
    assert(mat->ptr[i] <= nnz);
  }
  for(ptr_t j=0; j < nnz; ++j){
    assert(mat->ind[j] < ncols);
  }
}

void print_csr(csr_t *A, string name){
	cout<<"Matrix " << name <<": "<< A->nrows<<"x"<<A->ncols<<endl;

	cout << "ptr arr: ";
	for(idx_t i=0; i <= A->nrows; ++i){
		cout<<A->ptr[i]<<" ";
	}
	cout<<endl;
	cout << "ind arr: ";
	for(idx_t i=0; i < A->nrows; ++i){
		for(ptr_t j=A->ptr[i]; j < A->ptr[i+1]; ++j){
			cout << A->ind[j]<<" ";
		}
	}
	cout<<endl;
	cout << "val arr: ";
	for(idx_t i=0; i < A->nrows; ++i){
		for(ptr_t j=A->ptr[i]; j < A->ptr[i+1]; ++j){
			cout << A->val[j]<<" ";
		}
	}
	cout<<endl<<endl;
}

void print_sparse_mat(csr_t *A, string name){
  
	cout<<"Matrix " << name <<": "<< A->nrows<<"x"<<A->ncols<<endl;
	val_t row[A->ncols] = {0};
	cout<<"[";
	for(idx_t i = 0; i < A->nrows; i++){
		row[A->ncols] = {0};
		for (ptr_t x = A->ptr[i]; x < A->ptr[i+1]; x++){
			row[(A->ind)[x]] = (A->val)[x];
		}
		cout<<"[";
		for(idx_t x = 0; x < A->ncols; x++){
			cout<<row[x];
			if(x != A->ncols -1){
				cout<<",";
			}
			row[x] = 0;
		}
	
		if (i == A->nrows-1){
			cout<<"]]"<<endl;
		}
		else{
			cout<<"],"<<endl;
		}
	}
	cout<<endl;
}


void sort_csr(csr_t *mat){
 
  #pragma omp parallel for
  for ( idx_t row = 0; row < mat->nrows; row++){
		idx_t n= (mat->ptr[row+1] - mat->ptr[row]);
		if (n > 1){
			pair <idx_t, val_t> pairt[n];
			for (idx_t i = 0; i <n; i++ ){
				idx_t idx = i+ (mat->ptr)[row];
				pairt[i].first =(mat->ind)[idx];
				pairt[i].second = (mat->val)[idx];
			}
			sort(pairt, pairt + n);
			for (idx_t i = 0; i < n; i++ ){   
				idx_t idx = i+ (mat->ptr)[row];
				(mat->ind)[idx] = pairt[i].first;
				(mat->val)[idx] = pairt[i].second;
			}
		}
  	}
}

/**
 * Multiply A and B (transposed given) and write output in C.
 * Note that C has no data allocations (i.e., ptr, ind, and val pointers are null).
 * Use `csr_t::reserve` to increase C's allocations as necessary.
 * @param A  Matrix A.
 * @param B The transpose of matrix B.
 * @param C  Output matrix
 */
idx_t block_sparse_mat_mult(csr_t * A, csr_t * B, csr_t *C, idx_t a_start, idx_t a_end){
	idx_t c_max_nnz = (a_end - a_start) * 10;
  	idx_t c_nnz = 0;
  	int ind_val_idx = 0;
  	int ptr_idx = 0;
	//val_t dense_A_vector[10000] ;
	val_t *dense_A_vector{new val_t[A->ncols]{}};

  	C->reserve(A->nrows, c_max_nnz);
  	C->ptr[ptr_idx] = 0;
  	ptr_idx++;
	for (idx_t r = a_start; r < a_end; r++){

		
/* 		if(r % 1000 == 0){
			cout<<"ThreadID:"<<omp_get_thread_num()<<","<<a_start<<"~"<<a_end<<endl;
		} */
		

		//Set A vector to all zeros
		for(idx_t i =0; i < A->ncols; i++){
			dense_A_vector[i] = 0;
		}

		//Create A vector for row r
		for(idx_t i = (A->ptr)[r]; i < (A->ptr)[r+1]; i++){
			dense_A_vector[(A->ind)[i]] = (A->val)[i];
		}		


		for (idx_t c = 0; c < B->nrows; c++){
		
			val_t val = 0;
			for(idx_t n = (B->ptr)[c]; n < (B->ptr)[c+1]; n++){
				val += dense_A_vector[(B->ind)[n]] * (B->val)[n];
			}

		
			//If Val is non-zero, added it to sub CSR
			if (val != 0){
				c_nnz += 1;
				if(c_nnz >= c_max_nnz){
					c_max_nnz = int(2*c_max_nnz);
					C->ind = (idx_t*) realloc(C->ind, sizeof(idx_t) * c_max_nnz);
					C->val = (val_t*) realloc(C->val, sizeof(val_t) * c_max_nnz);
				}
				C->ind[ind_val_idx] = c;
				C->val[ind_val_idx] = val;
				ind_val_idx ++;
			}
		}
		C->ptr[ptr_idx] = c_nnz;
		ptr_idx++;
  	}

	C->nrows = a_end - a_start;
	C->ncols = B->nrows;
  	return c_nnz;
}


void sparsematmult(csr_t * A, csr_t * B, csr_t *C)
{
  	// A:rxn   B:cxn   C:rxc

 	sort_csr(B);

	//print_sparse_mat(A, "A");
	//print_sparse_mat(B, "B");
	
  	int block_size = int(0.01 * A->nrows);
	if (block_size < 1){
		block_size = 1;
	}
  	int block_count = (A->nrows+block_size-1)/block_size; //Gives Ceiling of A->nrows/block_size
  	csr_t *sub_C_csr[block_count];
	int sub_C_nnz[block_count];

	int total_nnz = 0;
	#pragma omp parallel for reduction(+: total_nnz)
	for (idx_t r_start = 0; r_start < A->nrows; r_start+=block_size){
		idx_t r_end = r_start + block_size;
		if (r_end >= A->nrows){
			r_end = A->nrows;
		}
		auto C_sub = new csr_t();
		idx_t block_idx = r_start / block_size;
		sub_C_nnz[block_idx] = block_sparse_mat_mult(A, B, C_sub, r_start,r_end );
		sub_C_csr[block_idx] = C_sub;
		total_nnz += sub_C_nnz[block_idx];
	
	}

	C->reserve(A->nrows+1, total_nnz);
	C->nrows = A->nrows;
	C->ncols = B->nrows;
	C->ptr[0] = 0;

	int base_ptr = 0;
	int base_ind = 0;
  	for(int idx =0; idx < block_count; idx++){

		//Update ptr
		for(idx_t i = 1; i < sub_C_csr[idx]->nrows+1; i++ ){
			(C->ptr)[i+base_ptr] =  (sub_C_csr[idx]->ptr)[i] + (C->ptr)[base_ptr];
		}	
		base_ptr += sub_C_csr[idx]->nrows;
	
		//Update ind and val
    	for(int i =0; i < sub_C_nnz[idx]; i++){
			(C->ind)[i+base_ind] = (sub_C_csr[idx]->ind)[i];
			(C->val)[i+base_ind] = (sub_C_csr[idx]->val)[i];
    	}
		base_ind += sub_C_nnz[idx];
		free(sub_C_csr[idx]);
	}
	//print_sparse_mat(C,"C");
}


int main(int argc, char *argv[])
{
	if(argc < 4){
		cerr << "Invalid options." << endl << "<program> <A_nrows> <A_ncols> <B_ncols> <fill_factor> [-t <num_threads>]" << endl;
		exit(1);
	}
	int nrows = atoi(argv[1]);
	int ncols = atoi(argv[2]);
	int ncols2 = atoi(argv[3]);
	double factor = atof(argv[4]);
	int nthreads = 1;
	
	cout << "A_nrows: " << nrows << endl;
	cout << "A_ncols: " << ncols << endl;
	cout << "B_ncols: " << ncols2 << endl;
	cout << "factor: " << factor << endl;
	cout << "nthreads: " << nthreads << endl;

	/* initialize random seed: */
	srand (time(NULL));

	omp_set_num_threads(28);
	auto A = csr_t::random(nrows, ncols, factor);
	auto B = csr_t::random(ncols2, ncols, factor); // Note B is already transposed.
	test_matrix(A);
	test_matrix(B);
	cout<<"Built A and B"<<endl;

	int threadcount[10] = {1,2,4,8,12,14,16,20,24,28};

	string printString;
    string excel_print[10];
	string line;
	cout<<"Matrix: "<<nrows<<"x"<<ncols<<"*"<<ncols<<"x"<<ncols2<<", Factor: "<<factor<<endl;
	if(argc == 7 && strcasecmp(argv[5], "-t") == 0){
		nthreads = atoi(argv[6]);
		omp_set_num_threads(nthreads);
		cout << "nthreads: " << nthreads << endl;
		auto C = new csr_t();
		auto t1 = omp_get_wtime();
		sparsematmult(A, B, C);
		auto t2 = omp_get_wtime();
		std::stringstream ss; 
		ss<<setfill(' ')<< setprecision(4)<<(t2-t1)*1000000<<",";
		cout<<"Thread:"<<nthreads<<" Time: "<<ss.str()<<endl;;;
		cout << C->info("C") << endl;
		delete C;
	}
	else
	{

		// Note that C has no data allocations so far.
		auto C = new csr_t();
		for(int i = 0; i < 10; i++){
			C = new csr_t();
			//omp_set_num_threads(28);	
			//B = csr_t::random(ncols2, ncols, factor);
			//test_matrix(B);
			
			std::stringstream ss; 
			omp_set_num_threads(threadcount[i]);
			cout<<setfill(' ') << setw(10)<<threadcount[i]<<",";
			auto t1 = omp_get_wtime();
			sparsematmult(A, B, C);
			auto t2 = omp_get_wtime();
			ss<<setfill(' ') << setw(10)<< setprecision(4)<<(t2-t1)*1000000<<",";
			cout<<"Thread:"<<threadcount[i]<<", Time: "<<ss.str()<<endl;
			printString += ss.str();
			
		}
		cout<<endl;
		cout<<"Time(µs):     "+printString;
		cout<<endl;
		cout << C->info("C") << endl;
		delete C;
	} 

	cout << A->info("A") << endl;
	cout << B->info("B") << endl;
	

	delete A;
	delete B;

	return 0;
}
