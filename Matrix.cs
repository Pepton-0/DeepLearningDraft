using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft
{
    [DataContract]
    public class Matrix
    {
        [DataMember]
        private double[,] matrix;

        /// <summary>
        /// Vertical axis
        /// </summary>
        [DataMember]
        public int rows { get; private set; }

        /// <summary>
        /// Horizontal axis
        /// </summary>
        [DataMember]
        public int columns { get; private set; }

        public Matrix(int rows, int columns)
        {
            matrix = new double[rows, columns];
            this.rows = rows;
            this.columns = columns;
        }

        public Matrix(double[,] values)
        {
            this.rows = values.GetLength(0);
            this.columns = values.GetLength(1);
            Array.Copy(values, matrix =
                new double[rows, columns],
                values.Length);
        }

        public Matrix(Matrix other)
        {
            this.rows = other.rows;
            this.columns = other.columns;
            Array.Copy(other.matrix, this.matrix, other.matrix.Length);
        }

        private Matrix(double[] vector, bool vertical)
        {
            if (vertical)
            {
                this.rows = vector.Length;
                this.columns = 1;
                this.matrix = new double[rows, columns];
                for (int i = 0; i < rows; i++)
                {
                    this.matrix[i, 0] = vector[i];
                }
            }
            else
            {
                this.columns = vector.Length;
                this.rows = 1;
                this.matrix = new double[rows, columns];
                for (int i = 0; i < columns; i++)
                {
                    this.matrix[0, i] = vector[i];
                }
            }
        }

        /// <summary>
        /// Create copy of this matrix.
        /// </summary>
        /// <returns></returns>
        public Matrix Clone()
        {
            return new Matrix(this);
        }

        /// <summary>
        /// Filter matrix values using a function.<br/>
        /// </summary>
        /// <param name="func"></param>
        public void FilterFunc(Func<double, double> func)
        {
            for(int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    matrix[i, j] = func(matrix[i, j]);
                }
            }
        }

        public void Randomize()
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    matrix[i, j] = App.rand.NextDouble(); // 0(inclusive) ~ 1(exclusive)
                }
            }
        }

        /// <summary>
        /// Horizontal vector 
        /// </summary>
        /// <param name="vec"></param>
        /// <returns></returns>
        public static Matrix ToCVector(double[] vec)
        {
            return new Matrix(vec, false);
        }

        /// <summary>
        /// Vertical vector
        /// </summary>
        /// <param name="vec"></param>
        /// <returns></returns>
        public static Matrix ToRVector(double[] vec)
        {
            return new Matrix(vec, true);
        }

        public static Matrix operator+(Matrix a, Matrix b)
        {
            if(a.rows != b.rows || a.columns != b.columns)
            {
                throw new ArgumentException("Matrices must have the same dimensions for addition.");
            }

            var v = new Matrix(a.rows, a.columns);
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    v.matrix[i, j] = a.matrix[i, j] + b.matrix[i, j];
                }
            }

            return v;
        }

        public static Matrix operator-(Matrix a, Matrix b)
        {
            if (a.rows != b.rows || a.columns != b.columns)
            {
                throw new ArgumentException("Matrices must have the same dimensions for subtraction.");
            }
            var v = new Matrix(a.rows, a.columns);
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    v.matrix[i, j] = a.matrix[i, j] - b.matrix[i, j];
                }
            }
            return v;
        }

        public static Matrix operator*(Matrix a, Matrix b)
        {
            if (a.columns != b.rows)
            {
                throw new ArgumentException("Number of columns in the first matrix must match number of rows in the second matrix.");
            }
            var v = new Matrix(a.rows, b.columns);
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < b.columns; j++)
                {
                    for (int k = 0; k < a.columns; k++)
                    {
                        v.matrix[i, j] += a.matrix[i, k] * b.matrix[k, j];
                    }
                }
            }
            return v;
        }
    }
}
