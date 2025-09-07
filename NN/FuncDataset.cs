using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft
{
    public class FuncDataset : IDataset
    {
        private readonly Func<double, double> Func;
        private readonly int SampleNum;

        /// <summary>
        /// Func's arg must be [0,1]
        /// </summary>
        /// <param name="sampleNum"></param>
        /// <param name="func"></param>
        public FuncDataset(int sampleNum, Func<double, double> func)
        {
            this.SampleNum = sampleNum;
            this.Func = func;
        }

        public (Matrix input, Matrix desiredOutput) GetSample(int index, bool test)
        {
            if (index > SampleNum)
                throw new Exception("Index should be the same or less than SampleNum");

            var ratio = (double)index / SampleNum;
            var value = Func(ratio);

            return (new Matrix(new double[,] { { ratio } }, true), new Matrix(new double[,] { { value } }, true));
        }

        public int GetSampleCount(bool test)
        {
            return SampleNum;
        }
    }
}
