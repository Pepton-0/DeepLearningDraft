using CommunityToolkit.Mvvm.ComponentModel;
using DeepLearningDraft.Models;
using DeepLearningDraft.Models.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft.ViewModels
{
    public class MainWindowViewModel : ObservableObject
    {
        private readonly IConfigService Config;
        private readonly IConductor Conductor;

        public MainWindowViewModel(IConfigService Config, IConductor Conductor)
        {
            this.Config = Config;
            this.Conductor = Conductor;
        }
    }
}
