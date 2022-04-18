#include <chrono>
#include <cassert>
#include <mpi.h>
#include <omp.h>
#include "hybridge.h"
#define CHUNKSIZE 4

namespace Hybrid{
    void ForwardElimination(double *matrix, double *rhs, int rows, int columns, int rank, int size){
        for(int row = 0; row < rows-1; row++){
            int block_idx = row / size;
            int l_rows_start;
            // 1st condition, "row" is above the row that the current process is responsible for
            // in the current "size" size block. The process should start its loop from the current "size" size block.
            if((row%size)<rank){
                l_rows_start = block_idx*size + rank;
            }
            // 2nd condition, "row" is not above the row that the current process is responsible for
            // in the current "size" size block. The process should start its loop from the next "size" size block.
            else{
                l_rows_start = (block_idx+1)*size + rank;
            }
            // Extract Diagonal element
            int diag_idx = row*rows + row;
            double diag_elem = matrix[diag_idx];
            assert(diag_elem!=0);
            // We implement OMP here
            // Change the number of threads as needed
            #pragma omp parallel for num_threads(2)
            for (int lower_rows=l_rows_start; lower_rows<rows; lower_rows+=size){
                int below_diag_idx = lower_rows*rows + row;
                // Compute the factor
                double elimination_factor = matrix[below_diag_idx]/diag_elem;
                int element_idx;
                for (int column=row+1; column<columns; column++){
                    // Set the column index of the entry to be operated
                    element_idx = lower_rows*rows + column;
                    // Subtract the row
                    matrix[element_idx] -= elimination_factor*matrix[row*rows+column];
                }
                rhs[lower_rows] -= elimination_factor*rhs[row];
            } 
            // after the elimination, we need to broadcast the newest row+1 to all threads,
            // so as to get ready for the next round and  the process with rank 0 will finally 
            // keep the result of ForwardSubstitution
            int r = (row+1)%size;
            MPI_Bcast(matrix+(row+1)*rows, columns, MPI_DOUBLE, r, MPI_COMM_WORLD);
            MPI_Bcast(rhs+row+1, 1, MPI_DOUBLE, r, MPI_COMM_WORLD);
        }
    }


    void BackwardSubstitution(double *matrix, double *rhs, double*solution,
                int rows, int columns, int rank, int size){
        // Set the size of the block, each thread would be responsible for block-size columns
        
        int block = columns / size;
        
        for(int row=rows-1; row>=0; row--){
            // Used to store the sum of parameters that ready to be extracted from rhs in each block
            double partial_sum = 0.0;
            // Used to sum up partial_sum
            double total_sum = 0.0;
            int diag_idx = row*rows + row;
            // 1st condition,row is smaller then block*rank
            // All columns in this block should be operated
            if(row < block*rank){
                // Uncomment the following line to implement OMP
                #pragma omp parallel for num_threads(4)
                for (int column=block*rank; column<block*(rank+1); column++){
                    int element_idx = row*rows + column;
                    // Uncomment the following line to implement OMP
                    #pragma omp critical
                    partial_sum += matrix[element_idx]*solution[column];
                }
            }
            // 2nd condition, number of row is in the block that the current thread is responsible for
            // Only columns in thes block that are larger then row should be operated
            else if(row>= block*rank && row<block*(rank+1)){
                // Uncomment the following line to implement OMP
                #pragma omp parallel for num_threads(4)
                for (int column=row+1; column<block*(rank+1); column++){
                    int element_idx = row*rows + column;
                     // Uncomment the following line to implement OMP
                    #pragma omp critical
                    partial_sum += matrix[element_idx]*solution[column];
                }
            }
            // Sum up the partial_sum and store it in tota_sum in rank 0
            MPI_Reduce(&partial_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            // Calculate the solution of the current row in rank 0
            if(rank == 0){
                solution[row] = (rhs[row] - total_sum) / matrix[diag_idx];
            }
            // Broadcast the solution to all other threads because the solution might be used in the next round
            MPI_Bcast(&solution[row], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
}


void Hybrid::Solve(double *matrix, double *rhs, double *solution,
                int rows, int columns){
    int rank, size;
    // Obtain the rank of the current thread and the size of the communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

   
   // Broadcast the parameters from rank 0 to other threads
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Need to apply for space before broadcasting matrix, rhs and solution for rank>0
    if(rank){
        matrix = new double[rows*columns];
        rhs = new double[rows];
        solution = new double[rows];
    }

    MPI_Bcast(matrix, rows*columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(rhs, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Hybrid::ForwardElimination(matrix, rhs, rows, columns, rank, size);
    Hybrid::BackwardSubstitution(matrix, rhs, solution, rows, columns, rank, size);
    if(rank){
        delete[] matrix;
        delete[] rhs;
        delete[] solution;
    }
}
