using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepLearningDraft.NN;

namespace DeepLearningDraft
{
    public class HalfAdderDataset : IDataset
    {
        private Matrix[] inputs = new Matrix[4] // Subset of (A,B)
        {
            new Matrix(new double[,]{{ 0 }, { 0 } }, true),
            new Matrix(new double[,]{{ 0 }, { 1 } }, true),
            new Matrix(new double[,]{{ 1 }, { 0 } }, true),
            new Matrix(new double[,]{{ 1 }, { 1 } }, true),
        };

        private Matrix[] outputs = new Matrix[4] // Subset of (C,D) C = A XOR B, D = A AND B
        {
            new Matrix(new double[,]{{0},{0}}, true),
            new Matrix(new double[,]{{1},{0}}, true),
            new Matrix(new double[,]{{1},{0}}, true),
            new Matrix(new double[,]{{0},{1}}, true),
        };

        public (Matrix inputs, Matrix desiredOutputs) GetSample(int index, bool test)
        {
            return (inputs[index], outputs[index]);
        }

        public int GetSampleCount(bool test)
        {
            return 4;
        }
    }
}
