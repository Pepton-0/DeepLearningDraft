using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft
{
    public class NN
    {
        /// <summary>
        /// n: numbers of nodes including input and output layers.<br/>
        /// Index from 0 (hidden layer after input layer) to n-1(output layer). <br/>
        /// Weight list for each layer.<br/>
        /// Array index is layer index.<br/>
        /// Matrix row axis is source node index
        /// Matrix column axis is target node index.
        /// currentLayerOutputs : i+1 = activationFunc(layerWeights[i] * prevLayerOutputs + layerBiases[i])
        /// </summary>
        private readonly Matrix[] layerWeights;

        /// <summary>
        /// Index from 0(hidden layer after input layer) to n-1(output layer).
        /// </summary>
        private readonly Matrix[] layerBiases;

        /// <summary>
        /// Index from 0(hidden layer after input layer) to n-1(output layer).
        /// </summary>
        private readonly ActivationFunction[] activationFuncs;

        public NN(params IntFuncPair[] pairs)
        {
            if(pairs.Length < 2)
            {
                throw new ArgumentException("Neural network must have at least input, output layers.");
            }

            if(pairs.Any(p => p.Integer <= 0))
            {
                throw new ArgumentException("All layers must have at least one node.");
            }

            layerWeights = new Matrix[pairs.Length - 1];
            layerBiases = new Matrix[pairs.Length - 1];
            activationFuncs = new ActivationFunction[pairs.Length - 1];

            for (int i = 1; i < pairs.Length - 1; i++)
            {
                var prevPair = pairs[i- 1];
                var currentPair = pairs[i];
                var weightMatrix = new Matrix(prevPair.Integer, currentPair.Integer);
                weightMatrix.Randomize();
                var biasMatrix = Matrix.ToRVector(new double[currentPair.Integer]);
                biasMatrix.Randomize();
                layerWeights[i - 1] = weightMatrix;
                layerBiases[i - 1] = biasMatrix;
                activationFuncs[i - 1] = currentPair.Func;
            }
        }

        public NN(Matrix[] layerWeights, Matrix[] layerBiases, ActivationFunction[] activationFuncs)
        {
            this.layerWeights = layerWeights;
            this.layerBiases = layerBiases;
            this.activationFuncs = activationFuncs;
        }

        /// <summary>
        /// First matrices are layer weights, second matrices are layer biases.
        /// </summary>
        /// <returns></returns>
        public (Matrix[], Matrix[]) CloneMatrices()
        {
            return (layerWeights.Select(m => m.Clone()).ToArray(),
                    layerBiases.Select(m => m.Clone()).ToArray());
        }
    }
}
