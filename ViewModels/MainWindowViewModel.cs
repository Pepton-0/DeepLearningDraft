using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DeepLearningDraft.Models;
using DeepLearningDraft.Models.Services;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace DeepLearningDraft.ViewModels
{
    public class MainWindowViewModel : ObservableObject
    {
        private readonly IConfigService Config;
        private readonly IConductor Conductor;

        public string ResultText
        {
            get => _resultText;
            set
            {
                if (!EqualityComparer<string>.Default.Equals(_resultText, value))
                {
                    _resultText = value;
                    // PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ResultText)));
                    OnPropertyChanged(nameof(ResultText));
                }
            }
        }
        private string _resultText;

        private RelayCommand<object> _eraseCanvas;
        public IRelayCommand<object> EraseCanvas => _eraseCanvas ?? new RelayCommand<object>((arg) =>
        {
            var inkCanvas = arg as InkCanvas;
            inkCanvas.Strokes.Clear();
            Log.Line("Cleared all the strokes");
        }, (o) => true);

        private RelayCommand<object> _strokeUpdated;
        public IRelayCommand<object> StrokeUpdated => _strokeUpdated ?? new RelayCommand<object>((arg) =>
        {
            // from inkCanvas image to (28,28) image
            var inkCanvas = arg as InkCanvas;
            var dv = new DrawingVisual();
            var dc = dv.RenderOpen();

            // its size might be bigger than inkCanvas cuz drawing out of canvas
            // ignore the area out of canvas
            var bounds = inkCanvas.Strokes.GetBounds();

            Log.Line("Bounds:" + bounds);
            Log.Line($"InkCanvas{inkCanvas.ActualWidth}, {inkCanvas.ActualHeight}");
            var rtb = new RenderTargetBitmap((int)inkCanvas.Width, (int)inkCanvas.Height, 96, 96, PixelFormats.Pbgra32);
            var scanRect = new Rect(
                bounds.TopLeft, 
                new System.Windows.Size(
                    Math.Min(inkCanvas.ActualWidth - bounds.TopLeft.X, bounds.Width),
                    Math.Min(inkCanvas.ActualHeight - bounds.TopLeft.Y, bounds.Height)));
            var scaleCenter = bounds.TopLeft + new Vector(scanRect.Size.Width / 2f, scanRect.Size.Height / 2f);
            var scale = new Vector(rtb.PixelWidth / scanRect.Width, rtb.PixelHeight / scanRect.Height);
            //m.ScaleAt(scale.X, scale.Y, scaleCenter.X, scaleCenter.Y);
            rtb.Render(inkCanvas);
            dc.Close();

            using (var ms = new MemoryStream())
            {
                var encoder = new BmpBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(rtb));
                encoder.Save(ms);
                var bitmap = new System.Drawing.Bitmap(ms);
                /*
                for (int h = 0; h < bitmap.Height; h++)
                {
                    for (int w = 0; w < bitmap.Width; w++)
                    {
                        if (bitmap.GetPixel(w, h).G < 100)
                            Log.NativeLine("#");
                        else
                            Log.NativeLine(" ");
                    }
                    Log.NativeLine("\n");
                }*/

                using (var sampleBitmap = new Bitmap(bitmap, 29, 29))
                {
                    using (var graphic = Graphics.FromImage(sampleBitmap))
                    {
                        
                        
                        /*
                        Log.Line("This is graphic");
                        graphic.DrawImage(
                            bitmap,
                            new Rectangle(0, 0, 28, 28));*/
                        /*
                        for (int h = 0; h < sampleBitmap.Height; h++)
                        {
                            for (int w = 0; w < sampleBitmap.Width; w++)
                            {
                                if (sampleBitmap.GetPixel(w, h).GetBrightness() < 0.5f)
                                    Log.NativeLine("#");
                                else
                                    Log.NativeLine("0");
                            }
                            Log.NativeLine("\n");
                        }*/
                        Matrix debugMatrix = new Matrix(28, 28);
                        Matrix matrix = new Matrix(28 * 28, 1);
                        for (int w = 0; w < sampleBitmap.Width - 1; w++)
                        {
                            for(int h = 0; h < sampleBitmap.Height - 1; h++)
                            {
                                if (bitmap.Width <= w || bitmap.Height <= h)
                                    continue;
                                var pixel = sampleBitmap.GetPixel(w + 1, h + 1);
                                var value = Math.Abs((double)(pixel.GetBrightness() - 1f));
                                matrix[h * 28 + w, 0] = value;
                                debugMatrix[h, w] = value;
                            }
                        }
                        for (int h = 0; h < debugMatrix.Rows; h++)
                        {
                            for (int w = 0; w < debugMatrix.Columns; w++)
                            {
                                if (debugMatrix[h,w] < 0.5f)
                                    Log.NativeLine("#");
                                else
                                    Log.NativeLine("0");
                            }
                            Log.NativeLine("\n");
                        }
                        ResultText = Conductor.Scan(matrix).ToString();
                    }
                }
            }

            // TODO rtb to grayscale bitmap(28.28)
        }, o => true);

        public MainWindowViewModel(IConfigService Config, IConductor Conductor)
        {
            this.Config = Config;
            this.Conductor = Conductor;
        }
    }
}
