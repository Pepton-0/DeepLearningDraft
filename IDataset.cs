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
        (double[] input, double[] desiredOutput) GetSample(int index, bool test);
    }
}
