using System;

namespace DeepLearningDraft
{
    public class Mathf
    {
        /*
        public static double Linear(double x)
        {
            return x;
        }*/

        public static void Linear(Matrix x)
        {
            return;
        }

        /*
        public static double LinearDiff(double x)
        {
            return 1;
        }*/

        public static void LinearDiff(Matrix x)
        {
            x.Execute((d) => 1d); // TODO no need to use prev data
        }

        /*
        public static double Sigmoid(double x)
        {
            return 1d / (1d + Math.Exp(-x));
        }*/

        public static void Sigmoid(Matrix x)
        {
            x.Execute((d) => 1d / (1d + Math.Exp(-d)));
        }

        /*
        public static double SigmoidDiff(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }*/

        public static void SigmoidDiff(Matrix x)
        {
            Sigmoid(x);
            x.Execute((d) => d * (1d - d));
        }

        /*
        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }*/

        public static void ReLU(Matrix x)
        {
            x.Execute((d) =>
            {
                return Math.Max(0, d);
            });
        }

        /*
        public static double ReLUDiff(double x)
        {
            if (x > 0)
            {
                return 1;
            }
            else return 0; // when x <= 0
        }*/

        public static void ReLUDiff(Matrix x)
        {
            // when x == 0: diff = 0d
            x.Execute((d) => d > 0 ? 1d : 0d);
        }

        /// <summary>
        /// From the same size matrices: outputs & answers, calculate loss function
        /// Loss = 1/2 * Σ(i=0, i<outputs.Length){(output[i,0] - answer[i,0])^2}
        /// dL/dOutput[any,0] = output - answer
        /// </summary>
        /// <param name="output"></param>
        /// <param name="answer"></param>
        /// <returns></returns>
        public static double Loss_SumOfSquareError(Matrix output, Matrix answer)
        {
            var matrix = output - answer;
            matrix.Execute((d) => d * d);
            var f = Matrix.Fill1(1, matrix.Rows) * matrix;
            return 0.5d * f[0, 0];
        }


        public static Matrix LossDiff_SumOfSquareError(Matrix output, Matrix answer)
        {
            return output - answer;
        }

        /// <summary>
        /// loss = Σ(-answer(n,0)*log(output(n,0)))
        /// </summary>
        /// <param name="output"></param>
        /// <param name="answer"></param>
        /// <returns></returns>
        public static double Loss_CrossEntropy(Matrix output, Matrix answer)
        {
            var matrix = output.Clone();
            Softmax(matrix);
            matrix.Execute((d) => Math.Log(d) * -1d);
            matrix.HadamarProduct(answer);
            var cmatrix = Matrix.Fill1(1, matrix.Rows);
            return (cmatrix * matrix)[0, 0];
        }

        public static Matrix LossDiff_CrossEntropy(Matrix output, Matrix answer)
        {
            return output - answer;
        }

        /// <summary>
        /// rmatrix(r,0) = e^rmatrix(r,0) / Σe^rmatrix(n,0)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static void Softmax(Matrix x)
        {
            x.Execute((d) => Math.Exp(d));
            var cmatrix = Matrix.Fill1(x.Columns, x.Rows);
            double sigma = (cmatrix * x)[0, 0];
            x.Execute((d) => d / sigma);
        }


        public static double Differential(Func<double, double> f, double x)
        {
            var from = f(x);
            var h = 1d;
            var prevDiff = 0d;
            var nextDiff = double.PositiveInfinity;
            do
            {
                h /= 2d;
                prevDiff = nextDiff;
                nextDiff = (f(x + h) - from) / h;

            } while (!Approximately(prevDiff, nextDiff));

            return nextDiff;
        }

        public static bool Approximately(double a, double b, double threshold = 0.00001d)
        {
            return Math.Abs(a - b) <= threshold;
        }
    }
}