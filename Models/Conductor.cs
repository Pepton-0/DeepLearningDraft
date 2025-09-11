using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft.Models
{
    public interface IConductor
    {
        int Scan(Matrix image);
    }

    public class Conductor : IConductor
    {
        private NN nn;

        public Conductor()
        {
            nn = NN.CreateFromFileOrNew("nn.xml", 8,
                LossFunction.CrossEntropy,
                new IntFuncPair(28 * 28, ActivationFunction.ReLu),
                new IntFuncPair(512, ActivationFunction.ReLu),
                new IntFuncPair(128, ActivationFunction.ReLu),
                new IntFuncPair(10, ActivationFunction.Linear));
        }

        public int Scan(Matrix image)
        {
            var matrix = nn.Calculate(image);
            matrix.Dump();
            return matrix.MaxCell().r;
        }
    }
}
