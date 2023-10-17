import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Container, Typography, Button } from '@material-ui/core';
import ErrorOutlineIcon from '@material-ui/icons/ErrorOutline';

const useStyles = makeStyles((theme) => ({
    root: {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        textAlign: 'center',
        backgroundColor: theme.palette.grey[200],
    },
    icon: {
        fontSize: 120,
        color: theme.palette.error.main,
    },
    title: {
        marginBottom: theme.spacing(2),
    },
}));

const ErrorPage = () => {
    const classes = useStyles();

    return (
        <Container className={classes.root}>
            <ErrorOutlineIcon className={classes.icon} />
            <Typography variant="h1" className={classes.title}>
                404
            </Typography>
            <Typography variant="subtitle1">
                Oops! The page you're looking for does not exist.
            </Typography>
            <Button variant="contained" color="primary" href="/">
                Go to Home
            </Button>
        </Container>
    );
};

export default ErrorPage;
