window.onload = () => {
    const navLinks = document.getElementsByClassName('nav-link');
    if (navLinks) {
      for (const element of navLinks) {
        element.classList.remove('active');
        if (element.href === window.location.href) {
          element.classList.add('active');
        }
      }
    }
  }

  