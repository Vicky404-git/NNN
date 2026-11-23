#include <iostream>
#include <vector>

using namespace std;

// Give this complex type a shorter name
using Matrix = vector<vector<int>>;


// FUNCTION 1: Take matrix input
Matrix inputMatrix(int rows, int cols) {

    // Create a matrix with the given dimensions
    // Initially, every element is 0
    Matrix matrix(rows, vector<int>(cols, 0));

    cout << "Enter matrix elements:\n";

    // Read each element
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cin >> matrix[i][j];
        }
    }

    return matrix;
}


// FUNCTION 2: Multiply two matrices
Matrix multiply(const Matrix& A, const Matrix& B) {

    // Get matrix dimensions
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();

    // Result dimensions:
    // Rows of A x Columns of B
    Matrix result(rowsA, vector<int>(colsB, 0));

    // First loop: Select a row from A
    for (int i = 0; i < rowsA; i++) {

        // Second loop: Select a column from B
        for (int j = 0; j < colsB; j++) {

            // Third loop: Multiply corresponding
            // elements and accumulate their sum
            for (int k = 0; k < colsA; k++) {

                result[i][j] += A[i][k] * B[k][j];

            }
        }
    }

    return result;
}


// FUNCTION 3: Print a matrix
void printMatrix(const Matrix& matrix) {

    for (const auto& row : matrix) {

        for (int value : row) {
            cout << value << "\t";
        }

        cout << '\n';
    }
}


// MAIN FUNCTION
int main() {

    int rowsA, colsA;
    int rowsB, colsB;

    // Get dimensions of Matrix A
    cout << "Enter rows and columns of Matrix A: ";
    cin >> rowsA >> colsA;

    // Get dimensions of Matrix B
    cout << "Enter rows and columns of Matrix B: ";
    cin >> rowsB >> colsB;

    // Matrix multiplication condition:
    // Columns of A must equal Rows of B
    if (colsA != rowsB) {

        cout << "Matrix multiplication is not possible!\n";

        return 1;
    }

    // Input matrices
    cout << "\nMatrix A:\n";
    Matrix A = inputMatrix(rowsA, colsA);

    cout << "\nMatrix B:\n";
    Matrix B = inputMatrix(rowsB, colsB);

    // Perform multiplication
    Matrix result = multiply(A, B);

    // Display result
    cout << "\nResult of A x B:\n";
    printMatrix(result);

    return 0;
}

