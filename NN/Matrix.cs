using MathNet.Numerics.LinearAlgebra;
using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using InternalMatrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

namespace DeepLearningDraft
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
                Log.NativeLine("    ");
                for (int k = 0; k < Columns; k++)
                {
                    Log.NativeLine(this[j, k].ToString("+000.000;-000.000;0000.000") + " ");
                }
                Log.NativeLine("\n");
            }
        }

        /// <summary>
        /// Re-plug in each value with the same function which uses the value
        /// </summary>
        /// <param name="func"></param>
        public virtual void Execute(Func<double, double> func)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    this[i, j] = func(this[i, j]);
                }
            }
        }

        /// <summary>
        /// Fill each value with return value of the func whic refers row and column
        /// </summary>
        /// <param name="func"></param>
        public virtual void FillFunc(Func<int, int, double> func)
        {
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    this[r, c] = func(r, c);
                }
            }
        }

        /// <summary>
        /// Fill all the values with random values between -1(exclusive) and 1(exclusive).<br/>
        /// </summary>
        public void Randomize()
        {
            FillFunc((r, c) =>
            {
                var d = NN.rand.NextDouble();
                if (NN.rand.Next() % 2 == 0)
                    d *= -1d;
                return d;
            }); // 0(inclusive) ~ 1(exclusive)
        }

        public byte[] ToByte1DimArr()
        {
            int sizeOfDouble = sizeof(double);
            var bytes = new byte[sizeOfDouble * Rows * Columns];
            for(int r = 0; r < Rows; r++)
            {
                for(int c = 0; c < Columns; c++)
                {
                    var partialBytes = BitConverter.GetBytes(this[r, c]);
                    partialBytes.CopyTo(bytes, (r * Columns + c) * sizeOfDouble);
                }
            }

            return bytes;
        }

        public static Matrix FromByte1DimArr(int rows, int columns, byte[] bytes)
        {
            var matrix = new Matrix(rows, columns);
            int sizeOfDouble = sizeof(double);
            matrix.FillFunc((r, c) =>
            {
                return BitConverter.ToDouble(bytes, sizeOfDouble * (r * columns + c));
            });
            return matrix;
        }

        public (int r, int c, double d) MaxCell()
        {
            int row = 0, col = 0;
            double d = double.NegativeInfinity;
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    var value = this[r, c];
                    if (value > d)
                    {
                        d = value;
                        row = r;
                        col = c;
                    }
                }
            }

            return (row, col, d);
        }

        public virtual Matrix Transpose()
        {
            var transposed = new Matrix(Columns, Rows);

            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    transposed[c, r] = this[r, c];
                }
            }

            return transposed;
        }

        public virtual void HadamarProduct(Matrix a)
        {
            if (this.Rows == a.Rows && this.Columns == a.Columns)
            {
                this.FillFunc((r, c) => this[r, c] * a[r, c]);
            }
            else
            {
                throw new Exception("a and b dont have the same rows and columns");
            }
        }

        public virtual void RunFuncForEachCell(Action<int, int, double> func)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
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
            for (int i = 0; i < matrix.Columns; i++)
            {
                for (int j = 0; j < top.Rows; j++)
                {
                    matrix[j, i] = top[j, i];
                }
                for (int j = 0; j < bottom.Rows; j++)
                {
                    matrix[j + top.Rows, i] = bottom[j, i];
                }
            }

            return matrix;
        }

        public virtual Matrix SelectColumn(int begin, int end)
        {
            if (begin > end || begin < 0 || end > this.Columns)
                throw new ArgumentException("begin and end are not correct args");

            var ripped = new Matrix(this.Rows, end - begin);

            for (int col = begin; col < end; col++)
            {
                for (int row = 0; row < this.Rows; row++)
                {
                    ripped[row, col - begin] = this[row, col];
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
        public static readonly MatrixBuilder<double> builder = InternalMatrix.Build;
        private readonly InternalMatrix matrix;

        public override double this[int row, int col]
        {
            get => matrix[row, col];
            set => matrix[row, col] = value;
        }

        public override int Rows { get; protected set; }
        public override int Columns { get; protected set; }

        public Matrix(Matrix other) : this(other.matrix.Clone()) { }

        public Matrix(int rows, int columns) : this(builder.Dense(rows, columns)) { }

        public Matrix(double[,] raw, bool thisValueIsNotUsedAnymore) : this(builder.DenseOfArray(raw)) { } 

        private Matrix(InternalMatrix arg)
        {
            matrix = arg;
            Rows = arg.RowCount;
            Columns = arg.ColumnCount;
        }

        public override void Execute(Func<double, double> func)
        {
            matrix.MapInplace(func);
        }

        public override void FillFunc(Func<int, int, double> func)
        {
            matrix.MapIndexedInplace((r, c, from) => func(r, c));
        }

        public override void HadamarProduct(Matrix a)
        {
            base.HadamarProduct(a);
        }

        public override void RunFuncForEachCell(Action<int, int, double> func)
        {
            matrix.MapIndexedInplace((r,c,from) => { func(r,c,from); return from; });
        }

        /// <summary>
        /// Copy the content of this matrix to a(rowBegin ~ rowBegin+this.Rows,columnBegin ~ this.Columns)
        /// </summary>
        /// <param name="rowBegin"></param>
        /// <param name="columnBegin"></param>
        /// <param name="a"></param>
        public void CopyTo(int rowBegin, int columnBegin, Matrix a)
        {
            a.matrix.SetSubMatrix(rowBegin, columnBegin, this.matrix);
        }

        public override Matrix Transpose()
        {
            // return base.Transpose();
            return new Matrix(matrix.Transpose());
        }

        public override Matrix Clone()
        {
            return new Matrix(this);
        }

        public override Matrix SelectColumn(int begin, int end)
        {
            var extracted = new Matrix(this.Rows, end - begin);
            matrix.SetSubMatrix(0, 0, 0, end - begin, extracted.matrix);
            return extracted;
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            return new Matrix(a.matrix + b.matrix);
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            return new Matrix(a.matrix - b.matrix);
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            return new Matrix(a.matrix * b.matrix);
        }
    }
}