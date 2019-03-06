import torch as tor
import torchvision
import torchvision.transforms as transforms


test_size = 4
#Creation du tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

#Import d'un jeu de test (sans Training)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#Testeur
testloader = tor.utils.data.DataLoader(testset, batch_size= test_size, shuffle=False, num_workers=2)
#Definition des classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class NNStat():
    def __init__(self):
        dataiter = iter(testloader)
        self.images, self.labels = dataiter.next()

    # Test model's accuracy
    def accuracy(self,nn):
        correct = 0
        total = 0
        with tor.no_grad():
            for data in testloader:
                images, labels = data
                outputs = nn(images)
                _, predicted = tor.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('\033[1mAccuracy:'+str(round((100 * correct / total),2))+'\033[0m')
        return (100 * correct / total)
    
    # Test model on nbim images
    def test(self,nn,nbim):
        outputs = nn(self.images)
        _, predicted = tor.max(outputs, 1)
        for j in range(nbim):
            sym = "\033[32m✓\033[0m" if self.labels[j] == predicted[j] else "\033[31m✗\033[0m"
            print("["+classes[self.labels[j]]+"]: "+classes[predicted[j]]+ " "+sym)

    # Test model accuracy per class
    def class_accuracy(self,nn):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with tor.no_grad():
            for data in testloader:
                images, labels = data
                outputs = nn(images)
                _, predicted = tor.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                    
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))