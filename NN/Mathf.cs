using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft.NN
{
    public class Mathf
    {
        public static double Sigmoid(double x)
        {
            return 1d / (1d + Math.Exp(-x));
        }

        public static double SigmoidDiff(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        public static double SigmoidDerivative(double x)
        {
            var fx = Sigmoid(x);
            return fx * (1d - fx);
        }

        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }

        public static double ReLUDiff(double x)
        {
            if (x > 0)
            {
                return 1;
            }
            else return 0; // when x <= 0
        }

        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        public static double ELU(double x, double alpha = 1.0)
        {
            return x >= 0 ? x : alpha * (Math.Exp(x) - 1);
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

        /// <summary>
        /// 最急降下法による極小値の計算
        /// </summary>
        /// <param name="f"></param>
        /// <param name="x">Initial position</param>
        /// <param name="n">Learning rate</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double GradientDecent(Func<double, double> f, double x, double n)
        {
            double before;
            var next = double.PositiveInfinity;
            int counter = 0;

            do
            {
                counter++;
                if (counter > 1000)
                {
                    throw new Exception("The calculation is too much long");
                }
                before = next;
                next = x - Differential(f, x) * n;
            } while (!Approximately(next, before));

            return next;
        }

        public static bool Approximately(double a, double b, double threshold = 0.00001d)
        {
            return Math.Abs(a - b) <= threshold;
        }
    }
}
