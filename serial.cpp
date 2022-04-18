//
// Created by shubham on 14.05.21.
//

#include <cassert>
#include <iostream>
#include <chrono>
#include "utility.h"
#include "serial.h"
namespace Serial{
void ForwardElimination(double *matrix, double *rhs, int rows, int columns){
    int diag_idx, lower_rows, below_diag_idx, element_idx;
    double diag_elem, elimination_factor;
    double time = 0.0;
    for(int row = 0; row < rows; row++){
        // Extract Diagonal element
        diag_idx = row*rows + row;
        diag_elem = matrix[diag_idx];
        assert(diag_elem!=0);
        
        auto start = std::chrono::steady_clock::now();
        for (lower_rows=row+1; lower_rows<rows; lower_rows++){
            below_diag_idx = lower_rows*rows + row;
            // Compute the factor
            elimination_factor = matrix[below_diag_idx]/diag_elem;
            for (int column=row+1; column<columns; column++){
                element_idx = lower_rows*rows + column;
                // subtract the row
                matrix[element_idx] -= elimination_factor*matrix[row*rows+column];
            }
            rhs[lower_rows] -= elimination_factor*rhs[row];
            // set below diagonal elements to 0
            matrix[below_diag_idx] = 0.;
        }
        auto end = std::chrono::steady_clock::now();
        float time_temp = std::chrono::duration<float>(end - start).count();
        time+=time_temp;
    }
    
    // float time = std::chrono::duration<float>(end - start).count();
    std::cout<<time<<"s"<<std::endl;
}

void BackwardSubstitution(double *matrix, double *rhs, double*solution, int rows, int columns){
    for(int row=rows-1; row>=0; row--){
        solution[row] = rhs[row];
        int diag_idx = row*rows + row;
        for (int column=row+1; column<columns; column++){
            int element_idx = row*rows + column;
            solution[row] -= matrix[element_idx]*solution[column];
        }
        solution[row] /= matrix[diag_idx];
    }
}
}
void Serial::Solve(double *matrix, double *rhs, double *solution,
                   int rows, int columns){

    Serial::ForwardElimination(matrix, rhs, rows, columns);
    Serial::BackwardSubstitution(matrix, rhs, solution, rows, columns);
}

void Serial::SerialSolve(int argc, char **argv, float &sequential_runtime, int rank){
    if(rank==0)
    {
        int rows, columns;
        double *matrixSeq, *rhsSeq, *solutionSeq;
        std::string matrix_name, rhs_name, ref_name;
        Utility::ParseFilesNames(argc, argv, matrix_name, rhs_name, &rows, &columns);
        auto start1 = std::chrono::steady_clock::now();
        matrixSeq = new double[(rows) * (columns)];
        rhsSeq = new double[(rows)];
        solutionSeq = new double[(columns)];
        Utility::InitializeArray(matrix_name, rhs_name,
                                 matrixSeq, rhsSeq, solutionSeq, rows, columns);

        auto start = std::chrono::steady_clock::now();
        Serial::Solve(matrixSeq, rhsSeq, solutionSeq, rows, columns);
        auto end = std::chrono::steady_clock::now();
        sequential_runtime = std::chrono::duration<float>(end - start).count();
        std::cout<<sequential_runtime<<"s"<<std::endl;
        // Utility::PrintSolution(solutionSeq, rows, columns);
        delete[] matrixSeq;
        delete[] rhsSeq;
        delete[] solutionSeq;
        auto end1 = std::chrono::steady_clock::now();
        float sequential_runtime1 = std::chrono::duration<float>(end1 - start1).count();
        std::cout<<sequential_runtime1<<"s"<<std::endl;
    }
}
