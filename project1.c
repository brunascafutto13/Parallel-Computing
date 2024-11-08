#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF 10000

typedef struct {
    int p;            //Number of processes          
    MPI_Comm comm;    // Global communicator
    MPI_Comm row_comm;  // Communicator for row
    MPI_Comm col_comm;  // Communicator for column
    int q;              //q^2 = p 
    int my_row;         
    int my_col;         
    int my_rank;        
} GRID_INFO_TYPE;

//Setup the grid for parallel processing
void Setup_grid(GRID_INFO_TYPE* grid) {
    int dimensions[2];
    int periods[2] = {0, 0};
    int coordinates[2];
    int varying_coords[2];

    // Get the number of processes and the rank of this process
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
    MPI_Comm_rank(MPI_COMM_WORLD, &grid->my_rank);

    // size of grid (q = sqrt(p))
    grid->q = (int) sqrt((double) grid->p);
    dimensions[0] = dimensions[1] = grid->q;

    // Create a Cartesian grid communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 1, &(grid->comm));

    // Determine the row and column coordinates of this process in the grid
    MPI_Cart_coords(grid->comm, grid->my_rank, 2, coordinates);
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    // Create communicators for rows and columns
    varying_coords[0] = 0; varying_coords[1] = 1;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->row_comm));
    varying_coords[0] = 1; varying_coords[1] = 0;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->col_comm));
}

// Read the matrix from the input file and divide in submatrixes
void Read_matrix(int** submatrix, int* n, GRID_INFO_TYPE* grid) {
    int* global_matrix = NULL;
    int n_bar;

    // Read the matrix from the input file
    if (grid->my_rank == 0) {
        FILE* file = fopen("input.txt", "r");
        if (!file) {
            fprintf(stderr, "Erro ao abrir o arquivo de entrada\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fscanf(file, "%d", n); // Read the size of the matrix
        global_matrix = (int*) malloc((*n) * (*n) * sizeof(int));

        for (int i = 0; i < *n; i++) {
            for (int j = 0; j < *n; j++) {
                fscanf(file, "%d", &global_matrix[i * (*n) + j]);
                //If there is no path between two nodes (and it's not the diagonal), set the value to INF
                if (i != j && global_matrix[i * (*n) + j] == 0) {
                    global_matrix[i * (*n) + j] = INF;
                }
            }
        }
        fclose(file);
    }

    // Broadcast the size of the matrix to all processes
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate the size of the submatrix
    n_bar = *n / grid->q;
    
    // Allocate memory for the submatrix
    *submatrix = (int*) malloc(n_bar * n_bar * sizeof(int));

    // Create a datatype for the submatrix
    MPI_Datatype block_type;
    MPI_Datatype temp_type;

    // Create a vector type for the submatrix 
    MPI_Type_vector(n_bar, n_bar, *n, MPI_INT, &temp_type);
    MPI_Type_commit(&temp_type);

    // Create a resized type for the submatrix
    MPI_Type_create_resized(temp_type, 0, sizeof(int), &block_type);
    MPI_Type_commit(&block_type);

    int* sendcounts = NULL;
    int* displs = NULL;

    // If the process is the root, create the sendcounts and displs arrays
    if (grid->my_rank == 0) {
        sendcounts = (int*) malloc(grid->p * sizeof(int));
        displs = (int*) malloc(grid->p * sizeof(int));

        //Define the submatrixes format to be sent to each process
        for (int i = 0; i < grid->q; i++) {
            for (int j = 0; j < grid->q; j++) {
                //each process receives 1 submatrix
                sendcounts[i * grid->q + j] = 1; 
                //displacement of each submatrix in the global matrix - start of a submatrix for each process
                displs[i * grid->q + j] = i * n_bar * (*n) + j * n_bar; 
            }
        }
    }
    // Scatter the global matrix

    MPI_Scatterv(global_matrix, sendcounts, displs, block_type,
                 *submatrix, n_bar * n_bar, MPI_INT,
                 0, grid->comm);

    if (grid->my_rank == 0) {
        free(global_matrix);
        free(sendcounts);
        free(displs);
    }

    MPI_Type_free(&temp_type);
    MPI_Type_free(&block_type);
}

// Min-plus multiplication
void Min_plus_multiply(int* A, int* B, int* C, int n_bar, int my_rank) {
    for (int i = 0; i < n_bar; i++) {
        for (int j = 0; j < n_bar; j++) {
            int min_val = INF;
            for (int k = 0; k < n_bar; k++) {
                //  If the value is INF, there is no path between the nodes -> continue
                if (A[i * n_bar + k] == INF || B[k * n_bar + j] == INF) 
                    continue;
                
                // Calculate the sum of the values
                int sum = A[i * n_bar + k] + B[k * n_bar + j];
                if (sum < min_val) {
                    min_val = sum;
                }
            }
            C[i * n_bar + j] = min_val; // Store the minimum value
        }
    }
}

// Fox's algorithm -> Parallel matrix multiplication of processes in a grid
void Fox_algorithm(int* submatrix, int n, int n_bar, GRID_INFO_TYPE* grid) {

    //Allocate memory for the matrices
    int* A = (int*)malloc(n_bar * n_bar * sizeof(int));
    int* B = (int*)malloc(n_bar * n_bar * sizeof(int));
    int* C = (int*)malloc(n_bar * n_bar * sizeof(int));
    int* temp = (int*)malloc(n_bar * n_bar * sizeof(int));
    
    // Initialize A and B with the submatrix
    memcpy(A, submatrix, n_bar * n_bar * sizeof(int));
    memcpy(B, submatrix, n_bar * n_bar * sizeof(int));
    
    // Initioalize C (Result matrix) with INF 
    for (int i = 0; i < n_bar * n_bar; i++) {
        C[i] = INF;
    }

    //Make q interations
    for (int stage = 0; stage < grid->q; stage++) {

        //Define the source of the broadcast
        int bcast_source = (grid->my_row + stage) % grid->q;
        
        //The process responsible for the broadcast sends the submatrix to the other processes
        if (grid->my_col == bcast_source) {
            memcpy(temp, A, n_bar * n_bar * sizeof(int));
        }
        
        //Broadcast the submatrix to the other processes in the same row
        MPI_Bcast(temp, n_bar * n_bar, MPI_INT, bcast_source, grid->row_comm);
        
        //  Multiply the submatrices
        int* temp_result = (int*)malloc(n_bar * n_bar * sizeof(int));
        Min_plus_multiply(temp, B, temp_result, n_bar, grid->my_rank);
        
        // Atualize the result matrix   
        for (int i = 0; i < n_bar * n_bar; i++) {
            if (temp_result[i] < C[i]) {
                C[i] = temp_result[i];
            }
        }
        
        free(temp_result);
        
        // Rotate the submatrix B in the column direction
        int src = (grid->my_row + 1) % grid->q;
        int dest = (grid->my_row - 1 + grid->q) % grid->q;
        
        MPI_Status status;
        //Send and receive the submatrix
        MPI_Sendrecv_replace(B, n_bar * n_bar, MPI_INT,
                            dest, 0, src, 0,
                            grid->col_comm, &status);
    }

    memcpy(submatrix, C, n_bar * n_bar * sizeof(int));

    free(A);
    free(B);
    free(C);
    free(temp);
}



void Repeated_squaring(int* submatrix, int n, GRID_INFO_TYPE* grid) {
    int n_bar = n / grid->q;
    //Calculate the number of interactions necessary to calculate the shortest path
    int max_iterations = (int)ceil(log2(n));
    
    for (int i = 0; i < max_iterations; i++) {
        Fox_algorithm(submatrix, n, n_bar, grid);
        MPI_Barrier(grid->comm);
    }
}

// Print the result
void Print_result(int* submatrix, int n, GRID_INFO_TYPE* grid) {
    int n_bar = n / grid->q;
    int* global_matrix = NULL; //Create new matrix to store the result
    
    if (grid->my_rank == 0) {
        global_matrix = (int*)malloc(n * n * sizeof(int));
    }


    MPI_Datatype block_type;
    MPI_Datatype temp_type;
    MPI_Type_vector(n_bar, n_bar, n, MPI_INT, &temp_type);
    MPI_Type_commit(&temp_type);
    MPI_Type_create_resized(temp_type, 0, sizeof(int), &block_type);
    MPI_Type_commit(&block_type);

    // Create the recvcounts and displs arrays for the gather operation
    int* recvcounts = NULL;
    int* displs = NULL;
    if (grid->my_rank == 0) {
        recvcounts = (int*)malloc(grid->p * sizeof(int));
        displs = (int*)malloc(grid->p * sizeof(int));
        for (int i = 0; i < grid->q; i++) {
            for (int j = 0; j < grid->q; j++) {
                recvcounts[i * grid->q + j] = 1;
                displs[i * grid->q + j] = i * n_bar * n + j * n_bar;
            }
        }
    }

    // Gather the submatrixes in the global matrix
    MPI_Gatherv(submatrix, n_bar * n_bar, MPI_INT,
                global_matrix, recvcounts, displs, block_type,
                0, grid->comm);

    if (grid->my_rank == 0) {
        printf("\nMatriz resultado (menores caminhos):\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (global_matrix[i * n + j] >= INF)
                    printf("0 "); //If there is no path between two nodes, print 0
                else
                    printf("%d ", global_matrix[i * n + j]);
            }
            printf("\n");
        }
        free(global_matrix);
        free(recvcounts);
        free(displs);
    }

    MPI_Type_free(&temp_type);
    MPI_Type_free(&block_type);
}

int main(int argc, char* argv[]) {
    int n;
    GRID_INFO_TYPE grid;
    int* submatrix;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    Setup_grid(&grid);

    // Check if the number of processes is a perfect square
    if (grid.p != grid.q * grid.q) {
        if (grid.my_rank == 0) {
            printf("Erro: O número de processos deve ser um quadrado perfeito.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Read the matrix from the input file and divide in submatrixes
    Read_matrix(&submatrix, &n, &grid);

    // Check if the size of the matrix is divisible by √P
    if (n % grid.q != 0) {
        if (grid.my_rank == 0) {
            printf("Erro: O tamanho da matriz (N=%d) deve ser divisível por √P (√P=%d)\n", n, grid.q);
        }
        free(submatrix);
        MPI_Finalize();
        return 0;
    }

    // Calculate the shortest path    
    Repeated_squaring(submatrix, n, &grid);
    //Gather and print the result
    Print_result(submatrix, n, &grid);

    free(submatrix);
    MPI_Finalize();
    return 0;
}