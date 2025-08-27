using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft
{
    public abstract class BaseMatrix
    {
        public abstract int Rows { get; protected set; }
        public abstract int Columns { get; protected set; }
        public abstract double this[int row, int column] { get; set; }
        public abstract Matrix Clone();
        public abstract void FilterFunc(Func<double, double> func);
        public abstract void Randomize();
        public abstract void Normalize();

        public void Dump()
        {
            for(int i = 0; i < Rows; i++)
            {
                for(int j = 0; j < Columns; j++)
                {
                    Console.Write(this[i, j] + " ");
                }
                Console.WriteLine();
            }
        }

        public static Matrix operator +(BaseMatrix a, BaseMatrix b)
        {
            if (a.Rows != b.Rows || a.Columns != b.Columns)
            {
                throw new ArgumentException("Matrices must have the same dimensions for addition.");
            }

            var v = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    v[i, j] = a[i, j] + b[i, j];
                }
            }

            return v;
        }

        public static Matrix operator -(BaseMatrix a, BaseMatrix b)
        {
            if (a.Rows != b.Rows || a.Columns != b.Columns)
            {
                throw new ArgumentException("Matrices must have the same dimensions for subtraction.");
            }
            var v = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    v[i, j] = a[i, j] - b[i, j];
                }
            }
            return v;
        }

        public static Matrix operator *(BaseMatrix a, BaseMatrix b)
        {
            if (a.Columns != b.Rows)
            {
                throw new ArgumentException("Number of columns in the first matrix must match number of rows in the second matrix.");
            }
            var v = new Matrix(a.Rows, b.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    for (int k = 0; k < a.Columns; k++)
                    {
                        v[i, j] += a[i, k] * b[k, j];
                    }
                }
            }
            return v;
        }
    }

    public class Matrix : BaseMatrix
    {
        private readonly double[,] matrix;
        public override int Rows { get; protected set; }
        public override int Columns { get; protected set; }

        public override double this[int row, int column]
        {
            get => matrix[row, column];
            set => matrix[row, column] = value;
        }

        public Matrix(int rows, int columns)
        {
            matrix = new double[rows, columns];
            this.Rows = rows;
            this.Columns = columns;
        }

        public Matrix(double[,] values, bool reuse = false)
        {
            this.Rows = values.GetLength(0);
            this.Columns = values.GetLength(1);

            if (reuse)
            {
                this.matrix = values;
            }
            else
            {
                Array.Copy(values, matrix =
                    new double[Rows, Columns],
                    values.Length);
            }
        }

        public Matrix(Matrix other)
        {
            this.Rows = other.Rows;
            this.Columns = other.Columns;
            this.matrix = new double[Rows, Columns]; 
            Array.Copy(other.matrix, this.matrix, other.matrix.Length);
        }

        private Matrix(double[] vector, bool vertical)
        {
            if (vertical)
            {
                this.Rows = vector.Length;
                this.Columns = 1;
                this.matrix = new double[Rows, Columns];
                
                for (int i = 0; i < Rows; i++)
                {
                    this[i, 0] = vector[i];
                }
            }
            else
            {
                this.Columns = vector.Length;
                this.Rows = 1;
                this.matrix = new double[Rows, Columns];
                
                for (int i = 0; i < Columns; i++)
                {
                    this[0, i] = vector[i];
                }
                //Array.Copy(vector, this.matrix, this.matrix.Length); this is not possible :(
            }
        }

        /// <summary>
        /// Create copy of this matrix.
        /// </summary>
        /// <returns></returns>
        public override Matrix Clone()
        {
            return new Matrix(this);
        }
        

        /// <summary>
        /// Filter matrix values using a function.<br/>
        /// </summary>
        /// <param name="func"></param>
        public override void FilterFunc(Func<double, double> func)
        {
            for(int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    this[i, j] = func(this[i, j]);
                }
            }
        }

        /// <summary>
        /// Fill all the values with random values between 0(inclusive) and 1(exclusive).<br/>
        /// </summary>
        public override void Randomize()
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    this[i, j] = App.rand.NextDouble(); // 0(inclusive) ~ 1(exclusive)
                }
            }
        }

        /// <summary>
        /// Horizontal vector 
        /// </summary>
        /// <param name="vec"></param>
        /// <returns></returns>
        public static Matrix FromCVector(double[] vec)
        {
            return new Matrix(vec, false);
        }

        /// <summary>
        /// Vertical vector
        /// </summary>
        /// <param name="vec"></param>
        /// <returns></returns>
        public static Matrix FromRVector(double[] vec)
        {
            return new Matrix(vec, true);
        }

        /// <summary>
        /// Return a matrix with same rows and columns that all values are divided by the norm of this matrix.
        /// </summary>
        /// <returns></returns>
        public override void Normalize()
        {
            double distance = 0;

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    var val = this[i, j];
                    distance += val * val;
                }
            }

            distance = Math.Sqrt(distance);
            FilterFunc((from) => from / distance);
        }
    }

    /// <summary>
    /// Row matrix that uses existing matrices without copying.
    /// </summary>
    public class SharedRMatrix : BaseMatrix
    {
        public override double this[int row, int column]
        {
            get
            {
                if(column != 0)
                {
                    throw new ArgumentOutOfRangeException("Column index must be 0 for SharedRMatrix.");
                }

                foreach(var m in matrices)
                {
                    if(row < m.Rows * m.Columns)
                    {
                        int r = row / m.Columns;
                        int c = row % m.Columns;
                        return m[r, c];
                    }
                    row -= m.Rows * m.Columns;
                }

                throw new ArgumentOutOfRangeException("Row index out of range for SharedRMatrix.");
            } 

            set 
            {
                if(column != 0)
                {
                    throw new ArgumentOutOfRangeException("Column index must be 0 for SharedRMatrix.");
                }

                foreach (var m in matrices)
                {
                    if (row < m.Rows * m.Columns)
                    {
                        int r = row / m.Columns;
                        int c = row % m.Columns;
                        m[r, c] = value;
                        return;
                    }
                    row -= m.Rows * m.Columns;
                }

                throw new ArgumentOutOfRangeException("Row index out of range for SharedRMatrix.");
            }
        }

        public override int Rows { get; protected set; }
        public override int Columns { get; protected set; }

        private readonly BaseMatrix[] matrices;

        public SharedRMatrix(params BaseMatrix[] matrices)
        {
            this.matrices = matrices;
            this.Rows = matrices.Sum(m => m.Rows * m.Columns);
            this.Columns = 1;
        }

        public override Matrix Clone()
        {
            var matrix = new Matrix(Rows, Columns);
            for(int i = 0; i < Rows; i++)
            {
                matrix[i, 0] = this[i, 0];
            }
            return matrix;
        }

        public override void FilterFunc(Func<double, double> func)
        {
            for(int i = 0; i < Rows; i++)
            {
                this[i, 0] = func(this[i, 0]);
            }
        }

        public override void Normalize()
        {
            throw new NotImplementedException();
        }

        public override void Randomize()
        {
            throw new NotImplementedException();
        }
    }
}
