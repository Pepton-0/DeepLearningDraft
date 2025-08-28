using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;

namespace DeepLearningDraft.NN
{
    public abstract class MatrixBase
    {
        public abstract int Rows { get; protected set; }
        public abstract int Columns { get; protected set; }
        public abstract double this[int row, int col] { get; set; }
        public abstract Matrix Clone();

        public void Dump()
        {
            Log.Line($"row:column={Rows}:{Columns}");
            for (int j = 0; j < Rows; j++)
            {
                Trace.Write("    ");
                for (int k = 0; k < Columns; k++)
                {
                    Trace.Write(this[j, k].ToString("000.000") + " ");
                }
                Trace.WriteLine("");
            }
        }

        /// <summary>
        /// Re-plug in each value with the same function which uses the value
        /// </summary>
        /// <param name="func"></param>
        public void Execute(Func<double, double> func)
        {
            for(int i = 0;i < Rows; i++)
            {
                for(int j = 0; j < Columns; j++)
                {
                    this[i, j] = func(this[i, j]);
                }
            }
        }

        /// <summary>
        /// Fill each value with return value of the func whic refers row and column
        /// </summary>
        /// <param name="func"></param>
        public void FillFunc(Func<int,int,double> func)
        {
            for(int r = 0; r <  Rows; r++)
            {
                for( int c = 0; c < Columns; c++)
                {
                    this[r, c] = func(r, c);
                }
            }
        }

        /// <summary>
        /// Fill all the values with random values between 0(inclusive) and 1(exclusive).<br/>
        /// </summary>
        public void Randomize()
        {
            FillFunc((r,c)=>App.rand.NextDouble()); // 0(inclusive) ~ 1(exclusive)
        }

        public Matrix Transpose()
        {
            var transposed = new Matrix(Columns, Rows);

            for(int r = 0; r < Rows; r++)
            {
                for(int  c = 0; c < Columns; c++)
                {
                    transposed[c, r] = this[r, c];
                }
            }

            return transposed;
        }

        public void HadamarProduct(Matrix a)
        {
            if(this.Rows == a.Rows && this.Columns == a.Columns)
            {
                this.FillFunc((r, c) => this[r, c] * a[r, c]);
            }
            else
            {
                throw new Exception("a and b dont have the same rows and columns");
            }
        }

        public void RunFuncForEachCell(Action<int,int, double> func)
        {
            for(int i = 0; i < Rows; i++)
            {
                for(int j = 0; j < Columns; j++)
                {
                    func(i, j, this[i, j]);
                }
            }
        }

        public static Matrix CombineRow(Matrix top, Matrix bottom)
        {
            if (top.Columns != bottom.Columns)
                throw new ArgumentException("Columns are not the same");

            var matrix = new Matrix(top.Rows + bottom.Rows, top.Columns);
            for(int i = 0; i < matrix.Columns; i++)
            {
                for(int j = 0; j < top.Rows; j++)
                {
                    matrix[j,i] = top[j,i];
                }
                for (int j = 0; j < bottom.Rows; j++)
                {
                    matrix[j + top.Rows, i] = bottom[j, i];
                }
            }

            return matrix;
        }

        public static Matrix SelectColumn(Matrix a, int begin, int end)
        {
            if (begin > end || begin < 0 || end > a.Columns)
                throw new ArgumentException("begin and end are not correct args");

            var ripped = new Matrix(a.Rows, end - begin);

            for(int col = begin; col < end; col++)
            {
                for(int row = 0; row < a.Rows; row++)
                {
                    ripped[row, col - begin] = a[row, col];
                }
            }

            return ripped;
        }

        public static Matrix CombineColumn(Matrix left, Matrix right)
        {
            if (left.Rows != right.Rows)
                throw new ArgumentException("Columns are not the same");

            var matrix = new Matrix(left.Rows, left.Columns + right.Columns);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < left.Columns; j++)
                {
                    matrix[j, i] = left[j, i];
                }
                for (int j = 0; j < right.Columns; j++)
                {
                    matrix[j + left.Columns, i] = right[j, i];
                }
            }

            return matrix;
        }

        public static Matrix Fill1(int rows, int columns)
        {
            var m = new Matrix(rows, columns);
            m.Execute((d) => 1);

            return m;
        }

        public static Matrix operator +(MatrixBase a, MatrixBase b)
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

        public static Matrix operator -(MatrixBase a, MatrixBase b)
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

        public static Matrix operator *(MatrixBase a, MatrixBase b)
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

    public class Matrix : MatrixBase
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
    }
}
