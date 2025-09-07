using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft
{
    public interface IDataset
    {
        int GetSampleCount(bool test);

        /// <summary>
        /// Input matrix should be (any, 1)
        /// Output matrix should be (any, 1)
        /// </summary>
        /// <param name="index"></param>
        /// <param name="test"></param>
        /// <returns></returns>
        (Matrix input, Matrix desiredOutput) GetSample(int index, bool test);
    }
}