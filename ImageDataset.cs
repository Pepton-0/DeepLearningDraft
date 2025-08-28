// #define TEST // for quick test
using System;
using System.Collections.Generic;
using System.IO;
using Path = System.IO.Path;
using DeepLearningDraft.NN;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.RightsManagement;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Markup;
using System.Windows.Shapes;

namespace DeepLearningDraft
{
    /// <summary>
    /// training data image: train-images.idx3-ubyte
    /// training data labe: train-labels.idx1-ubyte
    /// 
    /// test data image: t10k-images.idx3-ubyte
    /// test data label: t10k-labels.idx1-ubyte
    /// </summary>
    public class ImageDataset : IDataset
    {
        /// <summary>
        /// Store row*column byte pixel data.
        /// </summary>
        public readonly struct ImageBuffer
        {
            /// <summary>
            /// 0 to 1 pixel value.
            /// 0 means background(white), 1 means foreground(black).
            /// </summary>
            public readonly Matrix Pixel;

            private readonly int Width;

            public double this[int x, int y]
            {
                get => Pixel[Width * y + x, 0];
                set => Pixel[Width * y + x, 0] = value;
            }

            public ImageBuffer(int width, int height)
            {
                this.Width = width;
                this.Pixel = new Matrix(width * height, 1);
            }
        }

        public readonly struct ImageLabelPair
        {
            /// <summary>
            /// Matrix(10, 1)
            /// </summary>
            public readonly Matrix Label;
            public readonly ImageBuffer Image;

            public ImageLabelPair(byte label, ImageBuffer image)
            {
                Label = new Matrix(labelCount, 1);
                Label[label, 0] = 1.0; // One-hot encoding for label
                Image = image;
            }
        }

        private const int labelCount = 10; // MNIST has 10 labels from 0 to 9

        private readonly ImageLabelPair[] TrainingPair;

        private readonly ImageLabelPair[] TestPair;

        public ImageDataset(string dataDir)
        {
            string trainLabelPath = Path.Combine(dataDir, "train-labels.idx1-ubyte");
            string trainImagePath = Path.Combine(dataDir, "train-images.idx3-ubyte");
            string testLabelPath = Path.Combine(dataDir, "t10k-labels.idx1-ubyte");
            string testImagePath = Path.Combine(dataDir, "t10k-images.idx3-ubyte");

            Log.Line("Load training label");
            var trainLabels = LoadLabelFile(trainLabelPath);

            Log.Line("Load training image");
            var trainImages = LoadImageFile(trainImagePath);

            Log.Line("Load test label");
            var testLabels = LoadLabelFile(testLabelPath);

            Log.Line("Load test image");
            var testImages = LoadImageFile(testImagePath);

            TrainingPair = new ImageLabelPair[trainLabels.Length];
            for (int i = 0; i < trainLabels.Length; i++)
            {
                TrainingPair[i] = new ImageLabelPair(trainLabels[i], trainImages[i]);
            }

            TestPair = new ImageLabelPair[testLabels.Length];
            for (int i = 0; i < testLabels.Length; i++)
            {
                TestPair[i] = new ImageLabelPair(testLabels[i], testImages[i]);
            }
        }

        /// <summary>
        /// Get inputs(28 * 28, 1) and desired outputs(10, 1) of the sample
        /// </summary>
        /// <param name="index"></param>
        /// <param name="test"></param>
        /// <returns></returns>
        public (Matrix inputs, Matrix desiredOutputs) GetSample(int index, bool test)
        {
            var pair = test ? TestPair[index] : TrainingPair[index];

            (Matrix input, Matrix desiredOutput) result =
                (pair.Image.Pixel, pair.Label);
            return result;
        }

        public int GetSampleCount(bool test)
        {
            if(test)
            {
                return TestPair.Length;
            }
            else
            {
                return TrainingPair.Length;
            }
        }

        /// <summary>
        /// Load MNIST label file.
        /// The byte[] is label list from 0 to 9.
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        private static byte[] LoadLabelFile(string path)
        {
            using (FileStream fs = File.OpenRead(path))
            {
                Log.Line("Loading label...");
                int magicNumber = ReadInt32(fs); // Read the magic number
                Log.Line($"Magic number: {magicNumber}");

                int num = ReadInt32(fs); // Read the number of labels
#if TEST
                num = 10; // For testing, limit to 10 labels
#endif
                Log.Line($"Number of labels: {num}");
                byte[] labelBuffer = new byte[num];
                fs.Read(labelBuffer, 0, num);

                return labelBuffer;
            }
        }

        /// <summary>
        /// Load MNIST image file.
        /// Byte array is [index, rows, columns]
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        private static ImageBuffer[] LoadImageFile(string path)
        {
            using (FileStream fs = File.OpenRead(path))
            {
                Log.Line("Loading image...");
                int magicNumber = ReadInt32(fs); // Read the magic number
                Log.Line($"Magic number: {magicNumber}");
                int num = ReadInt32(fs); // Read the number of images
                Log.Line($"Number of images: {num}");
                int rows = ReadInt32(fs); // Read the number of rows
                int cols = ReadInt32(fs); // Read the number of columns
                Log.Line($"Image Size: {rows},{cols}");

                // from 0(background, white) to 255(foreground, black)
                byte[] buffer = new byte[num * rows * cols];

                ImageBuffer[] images = new ImageBuffer[num];
                fs.Read(buffer, 0, buffer.Length);

                // from start index(inclusive) to end index(exclusive)
                void StoreImage(int start, int end, int r, int c, ImageBuffer[] imgs, byte[] b)
                {
                    for (int i = start; i < end; i++)
                    {
                        imgs[i] = new ImageBuffer(cols, rows);
                        for (int j = 0; j < rows; j++)
                        {
                            for (int k = 0; k < cols; k++)
                            {
                                imgs[i][k, j] = b[i * rows * cols + j * cols + k] / 255.0d;
                            }
                        }
                    }
                }

                StoreImage(0, num, rows, cols, images, buffer);

                // Draw one example image

                int exampleIndex = 0;
                for(int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        double pixelValue = images[exampleIndex][j, i];
                        Console.Write(pixelValue > 0.5 ? "X" : " ");
                    }
                    Console.WriteLine();
                }

                return images;
            }
        }

        /// <summary>
        /// Read int32 as msb
        /// </summary>
        /// <param name="fs"></param>
        /// <returns></returns>
        private static int ReadInt32(FileStream fs)
        {
            int size = 4;
            int value = 0;
            byte[] buffer = new byte[size];
            fs.Read(buffer, 0, size);
            for (int i = 0; i < size; i++)
            {
                value |= buffer[i] << (size - i - 1) * 8;
            }

            return value;
        }
    }
}
    