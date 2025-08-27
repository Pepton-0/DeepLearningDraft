using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.TextFormatting;

namespace DeepLearningDraft
{
    public static class NMath
    {
        public static double Sigmoid(double x)
        {
            return 1d / (1d + Math.Exp(-x));
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
        
        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }
        
        public static double ELU(double x, double alpha = 1.0)
        {
            return x >= 0 ? x : alpha * (Math.Exp(x) - 1);
        }
        
        public static double ELUDerivative(double x, double alpha = 1.0)
        {
            return x >= 0 ? 1 : ELU(x, alpha) + alpha;
        }
        
        public static double Tanh(double x)
        {
            return Math.Tanh(x);
        }
        
        public static double TanhDerivative(double x)
        {
            var fx = Tanh(x);
            return 1 - fx * fx;
        }

        public static double Differential(Func<double, double> f, double x)
        {
            var from = f(x);
            double h = 1e-2;
            var prevDiff= 0d;
            double nextDiff = double.PositiveInfinity;
            do
            {
                h /= 2;
                prevDiff = nextDiff;
                nextDiff = (f(x + h) - from) / h;
            } while (!Approximately(prevDiff, nextDiff));

            return nextDiff;
        }

        public static bool Approximately(double a, double b, double threshold = 0.0001)
        {
            return Math.Abs(a - b) <= threshold;
        }
    }
}
